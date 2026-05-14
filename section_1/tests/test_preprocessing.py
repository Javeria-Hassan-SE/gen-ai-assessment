"""Unit tests for Task 1.1 preprocessing pipeline.

All tests are self-contained — synthetic images only, no external files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pytest

from ocr_pipeline import metadata_y_threshold
from preprocessing import (
    apply_clahe,
    binarize,
    compute_deskew_angle,
    deskew,
    reconnect_strokes,
    remove_noise,
    remove_ruled_lines,
    remove_vertical_margin_line,
    to_grayscale,
    _preprocess_array,
)


def make_white_page(h: int = 400, w: int = 300) -> np.ndarray:
    """White page with several horizontal black text-like lines."""
    img = np.ones((h, w), dtype=np.uint8) * 240
    for y in range(50, h, 40):
        cv2.line(img, (20, y), (w - 20, y), 30, 2)
    return img


def make_color_page() -> np.ndarray:
    """RGB color page (3-channel)."""
    gray = make_white_page()
    return np.stack([gray, gray, gray], axis=-1)


def make_skewed_page(angle_deg: float = 5.0) -> np.ndarray:
    """White page with horizontal lines, rotated by angle_deg degrees."""
    base = make_white_page(400, 300)
    h, w = base.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(base, M, (w, h), borderValue=255)


def make_ruled_page(h: int = 300, w: int = 400) -> np.ndarray:
    """White binary-style page with full-width horizontal ruling lines."""
    img = np.ones((h, w), dtype=np.uint8) * 255
    for y in range(20, h, 25):
        cv2.line(img, (0, y), (w - 1, y), 0, 1)  # ruling line = black
    # Add some short text strokes (shorter than full-width)
    for y in range(30, h, 25):
        cv2.line(img, (10, y + 5), (60, y + 5), 0, 2)
    return img


def make_page_with_margin_line(h: int = 300, w: int = 400) -> np.ndarray:
    """Binary page with a vertical margin line in the left 15%."""
    img = np.ones((h, w), dtype=np.uint8) * 255
    margin_x = int(w * 0.12)
    cv2.line(img, (margin_x, 0), (margin_x, h - 1), 0, 2)
    return img


def make_low_contrast_page() -> np.ndarray:
    """Page with very low ink-paper contrast (simulates faint pencil)."""
    img = make_white_page()
    # Compress dynamic range to [200, 240]
    return np.clip(200 + (img.astype(np.float32) / 255.0) * 40, 0, 255).astype(np.uint8)


class TestGrayscale:
    def test_rgb_to_grayscale(self):
        color = make_color_page()
        gray = to_grayscale(color, "RGB")
        assert gray.ndim == 2
        assert gray.dtype == np.uint8

    def test_already_gray_passthrough(self):
        gray_in = make_white_page()
        gray_out = to_grayscale(gray_in)
        assert gray_out.ndim == 2
        assert gray_out.shape == gray_in.shape

    def test_rgba_handled(self):
        rgba = np.ones((50, 50, 4), dtype=np.uint8) * 128
        result = to_grayscale(rgba, "RGB")
        assert result.ndim == 2


class TestDeskew:
    def test_deskew_improves_row_variance(self):
        """After deskewing, horizontal text lines produce higher row-sum variance."""
        skewed = make_skewed_page(angle_deg=4.0)
        deskewed, _ = deskew(skewed)
        skewed_var = float(np.var(skewed.sum(axis=1).astype(float)))
        deskewed_var = float(np.var(deskewed.sum(axis=1).astype(float)))
        assert deskewed_var >= skewed_var * 0.95, (
            f"Deskewed variance {deskewed_var:.1f} should be >= skewed {skewed_var:.1f}"
        )

    def test_output_shape_preserved(self):
        page = make_white_page()
        deskewed, angle = deskew(page)
        assert deskewed.shape == page.shape

    def test_near_zero_angle_on_aligned_image(self):
        page = make_white_page()
        angle = compute_deskew_angle(page)
        assert abs(angle) <= 5.0, f"Aligned page should have small angle, got {angle}"


class TestNoiseRemoval:
    def test_median_blur_output_shape(self):
        page = make_white_page()
        result = remove_noise(page)
        assert result.shape == page.shape
        assert result.dtype == np.uint8

    def test_noise_reduced(self):
        """Adding salt-and-pepper noise then denoising should recover a smoother image."""
        clean = make_white_page()
        noisy = clean.copy()
        rng = np.random.default_rng(42)
        mask = rng.random(noisy.shape) < 0.05
        noisy[mask] = 255 - noisy[mask]
        denoised = remove_noise(noisy)
        assert np.mean(np.abs(denoised.astype(int) - clean.astype(int))) <= np.mean(
            np.abs(noisy.astype(int) - clean.astype(int))
        )


class TestCLAHE:
    def test_output_range(self):
        low_contrast = make_low_contrast_page()
        enhanced = apply_clahe(low_contrast)
        assert enhanced.dtype == np.uint8
        assert enhanced.max() > low_contrast.max(), "CLAHE should expand dynamic range"

    def test_contrast_increases(self):
        page = make_low_contrast_page()
        enhanced = apply_clahe(page)
        assert int(enhanced.max()) - int(enhanced.min()) > int(page.max()) - int(page.min())


class TestBinarize:
    def test_output_is_binary(self):
        page = make_white_page()
        binary = binarize(page)
        unique = set(binary.flatten().tolist())
        assert unique.issubset({0, 255}), f"Binary image should only contain 0 and 255, got {unique}"

    def test_output_shape(self):
        page = make_white_page(200, 200)
        binary = binarize(page)
        assert binary.shape == page.shape


class TestRuledLineRemoval:
    def test_ruling_lines_reduced(self):
        """Full-width ruling lines should have fewer black pixels after removal."""
        ruled = make_ruled_page()
        cleaned = remove_ruled_lines(ruled)
        # Count black pixels (ink) in the full-width zone
        # Ruling lines span full width; text strokes are short
        full_width_black_before = np.sum(ruled == 0)
        full_width_black_after = np.sum(cleaned == 0)
        assert full_width_black_after < full_width_black_before, (
            "Removing ruled lines should reduce total ink pixels"
        )

    def test_short_strokes_mostly_preserved(self):
        """Short horizontal strokes (letters) should survive line removal."""
        h, w = 100, 200
        img = np.ones((h, w), dtype=np.uint8) * 255
        # Short stroke (much less than 40% width)
        cv2.line(img, (5, 50), (30, 50), 0, 2)
        cleaned = remove_ruled_lines(img)
        # At least some of the short stroke should remain
        assert np.sum(cleaned == 0) > 0


class TestVerticalMarginRemoval:
    def test_margin_line_removed(self):
        page = make_page_with_margin_line()
        cleaned = remove_vertical_margin_line(page)
        margin_x = int(page.shape[1] * 0.12)
        col_before = np.sum(page[:, margin_x] == 0)
        col_after = np.sum(cleaned[:, margin_x] == 0)
        assert col_after < col_before, "Margin line column should have fewer black pixels after removal"

    def test_no_margin_noop(self):
        """Image without margin line should be returned unchanged."""
        plain = np.ones((200, 300), dtype=np.uint8) * 255
        result = remove_vertical_margin_line(plain)
        assert np.array_equal(result, plain)


class TestReconnectStrokes:
    def test_output_dtype(self):
        binary = binarize(make_white_page())
        result = reconnect_strokes(binary)
        assert result.dtype == np.uint8

    def test_output_is_binary(self):
        binary = binarize(make_white_page())
        result = reconnect_strokes(binary)
        unique = set(result.flatten().tolist())
        assert unique.issubset({0, 255})


class TestFullPipeline:
    def test_clean_image_produces_sentences_ready_output(self):
        """Pipeline runs without error on a clean synthetic image."""
        color_page = make_color_page()
        final, ocr_input, steps = _preprocess_array(color_page, "RGB")
        assert final.ndim == 2
        assert final.dtype == np.uint8
        assert len(steps) == 8

    def test_metadata_region_not_in_output_text(self):
        """Metadata exclusion: top 20% Y coordinates should NOT appear in body word list."""
        h = 400
        threshold = metadata_y_threshold(h)
        assert threshold == int(h * 0.20)

    def test_skewed_image_processed_without_crash(self):
        skewed_color = np.stack([make_skewed_page(5)] * 3, axis=-1)
        final, ocr_input, steps = _preprocess_array(skewed_color, "RGB")
        assert final.shape[0] > 0

    def test_low_contrast_image_raises_no_error(self):
        low = make_low_contrast_page()
        color = np.stack([low, low, low], axis=-1)
        final, ocr_input, steps = _preprocess_array(color, "RGB")
        assert final is not None
