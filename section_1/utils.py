from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from config import DUPLICATE_HASH_THRESHOLD, BLUR_LAPLACIAN_THRESHOLD

try:
    from pdf2image import convert_from_path, convert_from_bytes
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    _PDF2IMAGE_AVAILABLE = False


def _find_poppler_path() -> str | None:
    # Checks for a local poppler/bin two levels up so no PATH change is needed.
    project_root = Path(__file__).parent.parent
    candidates = [
        project_root / "poppler" / "Library" / "bin",
        project_root / "poppler" / "bin",
        project_root / "poppler",
    ]
    for p in candidates:
        if (p / "pdftoppm.exe").exists():
            return str(p)
    return None


def pdf_to_images(pdf_path: str | None = None, pdf_bytes: bytes | None = None, dpi: int = 300) -> list[np.ndarray]:
    """Convert PDF pages to RGB numpy arrays via pdf2image."""
    if not _PDF2IMAGE_AVAILABLE:
        raise ImportError(
            "pdf2image is not installed. Run: pip install pdf2image\n"
            "Also install Poppler: extract the zip from\n"
            "  https://github.com/oschwartz10612/poppler-windows/releases\n"
            "  into  <project root>/poppler/  — no PATH change needed."
        )

    poppler_path = _find_poppler_path()

    if pdf_path:
        pil_pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    elif pdf_bytes:
        pil_pages = convert_from_bytes(pdf_bytes, dpi=dpi, poppler_path=poppler_path)
    else:
        raise ValueError("Provide either pdf_path or pdf_bytes.")

    return [np.array(p.convert("RGB")) for p in pil_pages]


def _perceptual_hash(gray: np.ndarray) -> np.ndarray:
    """Resize to 32x32 and return flattened float array for similarity comparison."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return small.flatten()


def _image_to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def images_are_similar(img_a: np.ndarray, img_b: np.ndarray) -> bool:
    """Return True when two page images appear to be the same content."""
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(_image_to_gray(img_b), (img_a.shape[1], img_a.shape[0]))
        img_a = _image_to_gray(img_a)
    else:
        img_a = _image_to_gray(img_a)
        img_b = _image_to_gray(img_b)

    ha = _perceptual_hash(img_a)
    hb = _perceptual_hash(img_b)
    correlation = float(np.corrcoef(ha, hb)[0, 1])
    return correlation > DUPLICATE_HASH_THRESHOLD


def detect_duplicate_pages(images: list[np.ndarray]) -> list[int]:
    """Return indices of pages that are likely duplicates of an earlier page."""
    duplicates: list[int] = []
    for i in range(1, len(images)):
        for j in range(i):
            if images_are_similar(images[i], images[j]):
                duplicates.append(i)
                break
    return duplicates


def compute_blur_score(gray: np.ndarray) -> float:
    """Laplacian variance — higher means sharper image."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_too_blurry(gray: np.ndarray) -> bool:
    return compute_blur_score(gray) < BLUR_LAPLACIAN_THRESHOLD


def rgb_to_display(img: np.ndarray) -> np.ndarray:
    """Ensure image is in a format Streamlit's st.image() accepts."""
    return img


def draw_ocr_boxes(
    image_rgb: np.ndarray,
    results: list[dict],
    metadata_threshold_y: int,
) -> np.ndarray:
    """Overlay EasyOCR bounding boxes on an RGB image copy."""
    pil = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for item in results:
        bbox = item["bbox"]
        pts = [(int(p[0]), int(p[1])) for p in bbox]

        if item["y1"] < metadata_threshold_y:
            color = (255, 215, 0, 160)
        elif item["low_confidence"]:
            color = (255, 20, 147, 160)
        else:
            color = (0, 200, 100, 120)

        draw.polygon(pts, fill=color)
        draw.line(pts + [pts[0]], fill=color[:3] + (255,), width=2)

    combined = Image.alpha_composite(pil, overlay).convert("RGB")
    return np.array(combined)


def default_pdf_path() -> str | None:
    """Return the Testing.pdf path if it exists next to section_1."""
    candidate = Path(__file__).parent.parent / "Testing.pdf"
    return str(candidate) if candidate.exists() else None
