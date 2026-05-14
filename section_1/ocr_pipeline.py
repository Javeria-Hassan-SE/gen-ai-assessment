"""Task 1.2 — EasyOCR extraction, reading-order reconstruction, sentence segmentation."""

from __future__ import annotations

import re

import numpy as np

from config import CONFIDENCE_THRESHOLD, LINE_Y_TOLERANCE, METADATA_HEIGHT_RATIO


OCR_ENGINES = {
    "easyocr": "EasyOCR    — CRAFT+CRNN, scene text, fast (default, no download)",
    "trocr":   "TrOCR ★    — handwriting specialist, ~400 MB one-time download",
}

_ENGINE_HF_MODEL = {
    "easyocr": None,
    "trocr":   "microsoft/trocr-base-handwritten",
}


def build_reader(engine: str = "easyocr", gpu: bool = False) -> dict:
    """Build and return an OCR reader bundle for the chosen engine.

    All three engines normalise output to the same word-dict format so the
    rest of the pipeline is engine-agnostic.

    engine choices:
      "easyocr"   — EasyOCR CRAFT+CRNN.  Already installed.
      "trocr"     — EasyOCR CRAFT for detection + microsoft/trocr-large-handwritten
                    for recognition.  Fine-tuned on the IAM handwriting benchmark —
                    far superior on cursive/child handwriting.
                    Requires: pip install transformers torch torchvision
      "paddleocr" — PaddleOCR PP-OCRv4 end-to-end.
                    Requires: pip install paddleocr paddlepaddle
    """
    try:
        import easyocr
    except ImportError:
        raise ImportError("easyocr not installed. Run: pip install easyocr")

    easyocr_reader = easyocr.Reader(["en"], gpu=gpu)

    if engine == "trocr":
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers torch torchvision"
            )
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        model.eval()
        device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
        model = model.to(device)
        return {
            "engine": "trocr",
            "easyocr": easyocr_reader,
            "trocr_processor": processor,
            "trocr_model": model,
            "device": device,
        }

    # default: easyocr
    return {"engine": "easyocr", "easyocr": easyocr_reader}



def run_ocr(image: np.ndarray, reader) -> list[dict]:
    """Dispatch to the active OCR engine and return normalised word-dicts."""
    engine = reader.get("engine", "easyocr") if isinstance(reader, dict) else "easyocr"
    if engine == "trocr":
        return _run_trocr_hybrid(image, reader)
    # easyocr (default)
    ocr_reader = reader["easyocr"] if isinstance(reader, dict) else reader
    raw = ocr_reader.readtext(image, detail=1, paragraph=False, mag_ratio=1.5)
    return _parse_easyocr_raw(raw)


def _parse_easyocr_raw(raw: list) -> list[dict]:
    results = []
    for bbox, text, conf in raw:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        results.append({
            "text": text.strip(),
            "confidence": round(float(conf), 4),
            "bbox": bbox,
            "x1": int(min(xs)), "y1": int(min(ys)),
            "x2": int(max(xs)), "y2": int(max(ys)),
            "low_confidence": float(conf) < CONFIDENCE_THRESHOLD,
        })
    return results


def _run_trocr_hybrid(image: np.ndarray, reader: dict) -> list[dict]:
    """CRAFT word detection → group into text lines → TrOCR on each line crop.

    TrOCR (microsoft/trocr-large-handwritten) was fine-tuned on IAM *line* images,
    not individual word crops.  Feeding it full text lines (each ~one row of writing)
    gives far better results than word-level crops because the decoder's language
    model context spans the whole line.
    """
    import torch
    from PIL import Image as PILImage

    easyocr_reader = reader["easyocr"]
    processor = reader["trocr_processor"]
    model = reader["trocr_model"]
    device = reader.get("device", "cpu")

    # Step 1: CRAFT word detection
    raw = easyocr_reader.readtext(image, detail=1, paragraph=False, mag_ratio=1.5)
    if not raw:
        return []

    word_boxes = []
    for bbox, easy_text, easy_conf in raw:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        word_boxes.append({
            "easy_text": easy_text, "easy_conf": float(easy_conf),
            "x1": int(min(xs)), "y1": int(min(ys)),
            "x2": int(max(xs)), "y2": int(max(ys)),
            "bbox": bbox,
        })
    word_boxes.sort(key=lambda w: (w["y1"], w["x1"]))

    # Step 2: group words into text lines by Y-proximity
    avg_h = float(np.mean([max(1, w["y2"] - w["y1"]) for w in word_boxes]))
    tolerance = max(10, avg_h * 0.6)

    lines: list[list[dict]] = [[word_boxes[0]]]
    for w in word_boxes[1:]:
        if abs(w["y1"] - lines[-1][0]["y1"]) <= tolerance:
            lines[-1].append(w)
        else:
            lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w["x1"])

    # Step 3: TrOCR recognition on each line crop
    results = []
    for line in lines:
        lx1 = min(w["x1"] for w in line)
        ly1 = min(w["y1"] for w in line)
        lx2 = max(w["x2"] for w in line)
        ly2 = max(w["y2"] for w in line)
        avg_conf = float(np.mean([w["easy_conf"] for w in line]))

        px, py = 8, 10
        cx1 = max(0, lx1 - px)
        cy1 = max(0, ly1 - py)
        cx2 = min(image.shape[1], lx2 + px)
        cy2 = min(image.shape[0], ly2 + py)
        crop = image[cy1:cy2, cx1:cx2]

        if crop.size == 0 or (cx2 - cx1) < 10:
            line_text = " ".join(w["easy_text"] for w in line)
        else:
            try:
                pil_crop = PILImage.fromarray(crop)
                pixel_values = processor(pil_crop, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    generated_ids = model.generate(pixel_values, max_new_tokens=64)
                line_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()
                if not line_text:
                    line_text = " ".join(w["easy_text"] for w in line)
            except Exception:
                line_text = " ".join(w["easy_text"] for w in line)

        # Step 4: emit one word-dict per token so downstream stays the same
        words = line_text.split()
        if not words:
            continue
        n = len(words)
        slot = max(1, (lx2 - lx1) // n)
        for i, word in enumerate(words):
            wx1 = lx1 + i * slot
            wx2 = min(lx2, wx1 + slot)
            results.append({
                "text": word,
                "confidence": round(avg_conf, 4),
                "bbox": [[wx1, ly1], [wx2, ly1], [wx2, ly2], [wx1, ly2]],
                "x1": wx1, "y1": ly1, "x2": wx2, "y2": ly2,
                "low_confidence": avg_conf < CONFIDENCE_THRESHOLD,
            })

    return results



def metadata_y_threshold(page_height: int) -> int:
    return int(page_height * METADATA_HEIGHT_RATIO)


def _has_metadata_pattern(words: list[dict]) -> bool:
    """Detect if a page has a metadata header (PS ###, class X-Y, or a date)."""
    full = " ".join(w["text"] for w in words)
    patterns = [
        r"\bPS\s*\d+",
        r"\bclass\s+\d",
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d+",
        r"\bDeceber\b",
    ]
    return any(re.search(p, full, re.IGNORECASE) for p in patterns)


def _parse_metadata_fields(words: list[dict], page_width: int, page_height: int) -> dict:
    """Parse metadata words into structured fields using X/Y position heuristics.

    Layout for Tadeo pages:
      Left half → school (row 1), class (row 2)
      Right half → student name (row 1), date (row 2)
      Horizontally centered, slightly lower → title
    """
    if not words:
        return {}

    threshold = metadata_y_threshold(page_height)
    mid_x = page_width * 0.50
    title_x_lo = page_width * 0.30
    title_x_hi = page_width * 0.70

    # Separate into left/right/centered columns by horizontal centroid
    left_words, right_words, center_words = [], [], []
    for w in words:
        cx = (w["x1"] + w["x2"]) / 2
        if title_x_lo < cx < title_x_hi and w["y1"] > threshold * 0.5:
            center_words.append(w)
        elif cx < mid_x:
            left_words.append(w)
        else:
            right_words.append(w)

    def _lines(wds: list[dict]) -> list[str]:
        """Group words into lines by Y proximity and join each line."""
        if not wds:
            return []
        sorted_w = sorted(wds, key=lambda w: w["y1"])
        lines: list[list[str]] = [[sorted_w[0]["text"]]]
        for w in sorted_w[1:]:
            if abs(w["y1"] - sorted_w[0]["y1"]) < 30:
                lines[-1].append(w["text"])
            else:
                lines.append([w["text"]])
                sorted_w[0] = w  # update reference for next comparison
        return [" ".join(l) for l in lines]

    left_lines = _lines(sorted(left_words, key=lambda w: w["y1"]))
    right_lines = _lines(sorted(right_words, key=lambda w: w["y1"]))
    center_lines = _lines(center_words)

    result: dict = {}
    if left_lines:
        result["school"] = left_lines[0]
    if len(left_lines) > 1:
        result["class"] = left_lines[1]
    if right_lines:
        result["student"] = right_lines[0]
    if len(right_lines) > 1:
        result["date"] = right_lines[1]
    if center_lines:
        result["title"] = center_lines[0]

    return result


def split_metadata_body(
    results: list[dict], page_height: int, page_width: int
) -> tuple[list[dict], list[dict], dict]:
    """Partition OCR results into (body, metadata_words, metadata_fields).

    If no metadata pattern is found in the top region, returns all words as
    body and metadata_fields = {} (handles page 2 which has no header).
    """
    threshold = metadata_y_threshold(page_height)
    top_words = [r for r in results if r["y1"] < threshold]
    body = [r for r in results if r["y1"] >= threshold]

    if not _has_metadata_pattern(top_words):
        # No header — all text is body (e.g. page 2 bus driver essay)
        return results, [], {}

    metadata_fields = _parse_metadata_fields(top_words, page_width, page_height)
    return body, top_words, metadata_fields



def reconstruct_reading_order(results: list[dict]) -> list[dict]:
    """Sort word boxes into top-to-bottom, left-to-right reading order."""
    if not results:
        return []

    heights = [max(1, r["y2"] - r["y1"]) for r in results]
    avg_h = float(np.mean(heights))
    tolerance = max(LINE_Y_TOLERANCE, avg_h * 0.5)

    sorted_y = sorted(results, key=lambda r: r["y1"])
    lines: list[list[dict]] = [[sorted_y[0]]]

    for item in sorted_y[1:]:
        if abs(item["y1"] - lines[-1][0]["y1"]) <= tolerance:
            lines[-1].append(item)
        else:
            lines.append([item])

    ordered: list[dict] = []
    for line in lines:
        ordered.extend(sorted(line, key=lambda r: r["x1"]))
    return ordered



def segment_sentences(text: str) -> list[str]:
    """Rule-based sentence segmentation using spaCy sentencizer.

    All neural components disabled — purely rule-based, no LLM.
    Handles edge cases like '8:00', '30sec', 'AAA A!' better than raw regex.
    """
    try:
        import spacy
    except ImportError:
        raise ImportError("spacy not installed. Run: pip install spacy")

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]



def _build_sentences(
    ordered_words: list[dict],
    low_conf_words: list[dict],
) -> list[dict]:
    """Segment ordered words into sentences matching reference output structure."""
    full_text = " ".join(w["text"] for w in ordered_words)
    raw_sentences = segment_sentences(full_text) if full_text.strip() else []

    lc_text_set = {w["text"].lower() for w in low_conf_words}
    tagged = []
    for idx, sent in enumerate(raw_sentences, start=1):
        sent_tokens = sent.lower().split()
        flagged_words = list(dict.fromkeys(t for t in sent_tokens if t in lc_text_set))
        tagged.append(
            {
                "id": idx,
                "text": sent,
                "low_confidence": len(flagged_words) > 0,
                "low_confidence_words": flagged_words,
            }
        )
    return tagged



def process_page(
    raw_image: np.ndarray,
    ocr_input: np.ndarray,
    reader,
    sample_id: str = "page",
    page_num: int = 0,
    is_duplicate: bool = False,
    blur_score: float = 0.0,
    deskew_angle: float = 0.0,
) -> dict:
    """Run the full OCR pipeline on a single preprocessed page.

    ocr_input must be a natural (non-binarized) RGB image — the deskewed
    original color image. EasyOCR is trained on natural images and produces
    garbage output when fed a binarized binary image.

    Returns a dict matching the reference output structure:
    {
        sample_id, metadata_excluded, sentences, page_low_confidence,
        _debug: { raw_ocr, body_words, full_text, stats, page_meta }
    }
    """
    h, w = ocr_input.shape[:2]

    ocr_results = run_ocr(ocr_input, reader)
    body_words, metadata_words, metadata_fields = split_metadata_body(ocr_results, h, w)
    ordered_body = reconstruct_reading_order(body_words)

    low_conf_words = [wd for wd in ordered_body if wd["low_confidence"]]
    sentences = _build_sentences(ordered_body, low_conf_words)
    page_low_confidence = any(s["low_confidence"] for s in sentences)

    full_text = " ".join(wd["text"] for wd in ordered_body)
    total_words = len(ordered_body)
    lc_ratio = len(low_conf_words) / total_words if total_words else 0.0

    return {
        "sample_id": sample_id,
        "metadata_excluded": metadata_fields,
        "sentences": sentences,
        "page_low_confidence": page_low_confidence,
        "_debug": {
            "raw_ocr": ocr_results,
            "metadata_words": metadata_words,
            "body_words": ordered_body,
            "full_text": full_text,
            "page_meta": {
                "page_num": page_num,
                "is_duplicate": is_duplicate,
                "blur_score": round(blur_score, 2),
                "deskew_angle": round(deskew_angle, 2),
                "low_conf_ratio": round(lc_ratio, 4),
                "metadata_y_threshold": metadata_y_threshold(h),
            },
            "stats": {
                "total_words": total_words,
                "low_confidence_words": len(low_conf_words),
                "total_sentences": len(sentences),
                "flagged_sentences": sum(1 for s in sentences if s["low_confidence"]),
            },
        },
    }
