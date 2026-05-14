"""Task 1.1 — Fixed sequential image preprocessing pipeline for handwritten scans."""

from __future__ import annotations

import cv2
import numpy as np

from config import (
    ADAPTIVE_BLOCK_SIZE,
    ADAPTIVE_C,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    DESKEW_ANGLE_RANGE,
    DESKEW_ANGLE_STEPS,
    MEDIAN_KERNEL_SIZE,
)



def to_grayscale(image: np.ndarray, color_space: str = "RGB") -> np.ndarray:
    """Convert color image to single-channel grayscale."""
    if len(image.shape) == 2:
        return image.copy()
    if image.shape[2] == 4:
        image = image[:, :, :3]
    code = cv2.COLOR_RGB2GRAY if color_space == "RGB" else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, code)



def compute_deskew_angle(gray: np.ndarray) -> float:
    """Detect dominant text angle via horizontal projection profile analysis.

    Downsamples for speed, applies Otsu binarization, then finds the angle
    whose row-sum variance is maximal (text lines maximally separated).

    Returns the correction angle in degrees.
    """
    h, w = gray.shape
    scale = min(1.0, 800 / max(h, w))
    small = cv2.resize(gray, (int(w * scale), int(h * scale))) if scale < 1.0 else gray.copy()

    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    sh, sw = binary.shape
    center = (sw // 2, sh // 2)
    best_angle, best_score = 0.0, -1.0

    for angle in np.linspace(DESKEW_ANGLE_RANGE[0], DESKEW_ANGLE_RANGE[1], DESKEW_ANGLE_STEPS):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (sw, sh), flags=cv2.INTER_LINEAR, borderValue=0)
        row_sums = rotated.sum(axis=1).astype(np.float64)
        score = float(np.var(row_sums))
        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle


def _rotate(image: np.ndarray, angle: float, border_value: int = 255) -> np.ndarray:
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=border_value)


def _rotate_color(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 3-channel image, filling with white (255,255,255)."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Detect and correct skew. Returns (corrected_image, angle_applied)."""
    angle = compute_deskew_angle(gray)
    return _rotate(gray, angle, border_value=255), angle



def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization.

    Applied BEFORE median filter so CLAHE can amplify faint strokes (e.g. light
    pencil on page 2) before noise suppression smooths them. This order matches
    the reference specification: CLAHE(3) then Median(4).
    """
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    return clahe.apply(gray)



def remove_noise(gray: np.ndarray) -> np.ndarray:
    """Median blur: preserves stroke edges while removing salt-and-pepper noise.

    Preferred over Gaussian because it does not blur letter-stroke edges that
    are critical for OCR character discrimination.
    """
    return cv2.medianBlur(gray, MEDIAN_KERNEL_SIZE)



def binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian thresholding — highest-impact step for handwriting OCR.

    Uses local neighbourhood thresholding to handle uneven illumination across
    the page (darker edges, varying ink pressure, shadow).

    Output: 0 = ink/text, 255 = paper background.
    """
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )



def remove_ruled_lines(binary: np.ndarray) -> np.ndarray:
    """Remove notebook ruling lines using horizontal morphological opening.

    Structuring element width = 40% of image width — captures full-page lines
    while leaving shorter letter strokes intact.

    binary convention: ink=0 (black), paper=255 (white).
    """
    h, w = binary.shape
    kernel_w = max(40, int(w * 0.40))

    inv = cv2.bitwise_not(binary)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel)

    d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    h_lines = cv2.dilate(h_lines, d_kernel)

    cleaned_inv = cv2.subtract(inv, h_lines)
    return cv2.bitwise_not(cleaned_inv)



def remove_vertical_margin_line(binary: np.ndarray) -> np.ndarray:
    """Detect and remove the vertical red margin line on notebook paper.

    Only examines the left 20% of the image — never affects text characters.
    Safe no-op if no tall vertical line is detected.
    """
    h, w = binary.shape
    kernel_h = max(40, int(h * 0.40))
    left_limit = int(w * 0.20)

    inv = cv2.bitwise_not(binary)
    left_region = inv[:, :left_limit]

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
    v_lines = cv2.morphologyEx(left_region, cv2.MORPH_OPEN, v_kernel)

    if v_lines.sum() == 0:
        return binary

    mask = np.zeros_like(inv)
    mask[:, :left_limit] = v_lines
    cleaned_inv = cv2.subtract(inv, mask)
    return cv2.bitwise_not(cleaned_inv)



def reconnect_strokes(binary: np.ndarray) -> np.ndarray:
    """Small morphological closing to bridge gaps introduced during binarization.

    2×1 horizontal kernel — reconnects only horizontally adjacent gaps to avoid
    merging characters on adjacent lines.
    """
    inv = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    return cv2.bitwise_not(closed)



def _preprocess_array(image: np.ndarray, color_space: str = "RGB") -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the complete preprocessing pipeline on an in-memory numpy array.

    Returns (final_binary, ocr_input, steps). ocr_input is the deskewed grayscale
    image — NOT binarized, because EasyOCR is trained on continuous-tone images.
    """
    steps: dict[str, tuple[np.ndarray, str]] = {}

    gray = to_grayscale(image, color_space)
    steps["1. Grayscale"] = (gray.copy(), "Color removed — all downstream processing on single channel")

    deskewed_gray, angle = deskew(gray)
    steps[f"2. Deskewed ({angle:+.1f}°)"] = (deskewed_gray.copy(), f"Projection-profile correction · angle={angle:+.2f}°")

    enhanced = apply_clahe(deskewed_gray)
    steps["3. CLAHE Enhanced"] = (enhanced.copy(), "Local contrast enhancement before noise removal (recovers faint pencil)")

    denoised = remove_noise(enhanced)
    steps["4. Noise Removed"] = (denoised.copy(), "Median blur (3×3) — preserves stroke edges, removes speckle")

    # OCR input: denoised CLAHE grayscale stacked to 3-channel.
    # Step 4 (after median blur) is cleaner than step 3 — speckle noise is removed
    # while stroke edges are preserved.  Still not binarized — OCR models expect
    # continuous-tone input, not binary.
    ocr_input = np.stack([denoised] * 3, axis=-1)

    binary = binarize(denoised)
    steps["5. Binarized"] = (binary.copy(), "Adaptive Gaussian threshold — handles uneven illumination")

    no_hlines = remove_ruled_lines(binary)
    steps["6. Lines Removed"] = (no_hlines.copy(), "Horizontal morphological opening removes ruled notebook lines")

    no_margin = remove_vertical_margin_line(no_hlines)
    steps["7. Margin Removed"] = (no_margin.copy(), "Vertical margin line detection + removal (left 20% only — extra step)")

    final = reconnect_strokes(no_margin)
    steps["8. Final (display only)"] = (final.copy(), "Binary for display — NOT used by EasyOCR (see ocr_input)")

    return final, ocr_input, steps



def preprocess_image(image_path: str) -> np.ndarray:
    """Load an image from disk and return a cleaned binarised H×W uint8 array."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    final, _, _ = _preprocess_array(img, color_space="BGR")
    return final
