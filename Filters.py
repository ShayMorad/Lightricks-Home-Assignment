"""
filters.py – *pure‑NumPy* image filters & adjustments used by the *edit‑image* CLI.

Key design points
-----------------
* **Stateless**: every function takes a ndarray and returns a fresh one.
* **Zero heavy deps**: only needs *NumPy* + *Matplotlib* (already used for
  I/O in the CLI), no OpenCV/Pillow.
* **Strict validation** – `_validate_image` accepts only grayscale `(H,W)`
  or RGB `(H,W,3)`.  An RGBA image is *silently* trimmed to RGB so you can
  load transparent PNGs and still proceed.
* **Per‑channel processing** – color operations never bleed across planes.
* **Box blur & Sobel** implemented from scratch using a tiny convolution
  helper; sharpening uses a hand‑rolled Gaussian kernel.
"""

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import os

__all__ = ["Filters"]


class Filters:
    """Namespace for all image‑processing helpers (no state kept)."""

    # ---------------------------------------------------------------------
    # Internal color‑space tags (int keeps them json‑serialisable if needed)
    # ---------------------------------------------------------------------
    _GRAYSCALE: int = 1  # (H, W)
    _RGB: int = 3  # (H, W, 3)
    _RGBA: int = 4  # (H, W, 4)

    # ---------------------------------------------------------------------
    # ───────────── VALIDATION & TYPE CONVERSION HELPERS ───────────────────
    # ---------------------------------------------------------------------
    @staticmethod
    def _validate_image(img: np.ndarray) -> int:
        """Guarantee *img* is grayscale or RGB; return the detected color‑space.

        *RGB images with a 4th channel (alpha) are **trimmed** to RGB* so the
        caller doesn’t crash on transparent PNGs.
        """
        if img.ndim == 2:
            return Filters._GRAYSCALE
        if img.ndim == 3 and img.shape[2] == 3:
            return Filters._RGB
        if img.ndim == 3 and img.shape[2] == 4:  # RGBA → return RGBA and trim back at '_to_float' if needed
            return Filters._RGBA
        raise ValueError(
            f"Unsupported image shape {img.shape}. "
            "Expected grayscale (H, W) or RGB (H, W, 3)."
        )

    # -------------------------- Type conversion -----------------------------

    @staticmethod
    def _to_float(img: np.ndarray) -> Tuple[np.ndarray, np.dtype, int]:
        """Return float32 copy of `img`, original dtype, and color‑space tag."""
        cs = Filters._validate_image(img)
        if cs == Filters._RGBA:
            img = img[..., :3]
        orig_dtype = img.dtype
        if img.dtype == np.uint8:
            f = img.astype(np.float32)
        else:
            f = img.astype(np.float32)
            if f.max() <= 1.0:  # If floats are in 0‑1 range, rescale so all ops are in 0‑255 domain
                f *= 255.0
        return f, orig_dtype, cs

    @staticmethod
    def _restore_type(img_f: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Clip back to [0,255] and cast to original dtype (float retains scale)."""
        img_c = np.clip(img_f, 0, 255)
        if dtype == np.uint8:
            return img_c.astype(np.uint8)
        if img_c.max() > 1.0 and dtype != np.uint8:  # keep floats in same 0‑1 or 0‑255 range
            img_c /= 255.0
        return img_c.astype(dtype)

    # ---------------------------------------------------------------------
    # ─────────────────────────── 2‑D CONVOLUTION ──────────────────────────
    # ---------------------------------------------------------------------

    @staticmethod
    def _convolve2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Manual convolution with reflect padding (single channel)."""
        kh, kw = kernel.shape
        # pad by half the kernel size in each direction because at most, the corner has half the kernel size
        # "missing" pixels beyond it.
        ph, pw = kh // 2, kw // 2
        # surround the image with extra pixels using reflect padding. helps with convolving extreme areas such as the
        # corners.
        padded = np.pad(channel, ((ph, ph), (pw, pw)), mode="reflect")
        # mathematical convolution requires flipping the kernel in both axes then applying the kernel by
        # multiplying it with the image.
        kern = np.flipud(np.fliplr(kernel))
        out = np.zeros_like(channel, dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                out += kern[i, j] * padded[i: i + channel.shape[0], j: j + channel.shape[1]]
        return out

    # ---------------------------------------------------------------------
    # ───────────────────────────── GAUSSIAN KERNEL ────────────────────────
    # ---------------------------------------------------------------------

    @staticmethod
    def _gaussian_kernel(radius: int) -> np.ndarray:
        """Return a normalised (2R+1)² Gaussian kernel with σ = R/2."""
        size = 2 * radius + 1
        sigma = radius / 2.0
        ax = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(ax, ax)
        k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        k /= k.sum()
        return k.astype(np.float32)

    # ---------------------------------------------------------------------
    # ─────────────────────────────  FILTERS  ──────────────────────────────
    # ---------------------------------------------------------------------

    @staticmethod
    def box_blur(image: np.ndarray, kx: int, ky: int) -> np.ndarray:
        """Mean blur with a *kx×ky* (Width x Height) rectangular kernel."""
        if kx <= 0 or ky <= 0:
            raise ValueError("Kernel dimensions must be positive integers")
        img_f, dtype, cs = Filters._to_float(image)
        kernel = np.ones((ky, kx), dtype=np.float32) / (kx * ky)
        if cs == Filters._GRAYSCALE:
            blurred = Filters._convolve2d(img_f, kernel)
        else:
            # blur each channel/plane separately... for c=0 (red), c=1(green), c=2(blue)
            blurred = np.stack([Filters._convolve2d(img_f[..., c], kernel) for c in range(3)], axis=-1, )
        return Filters._restore_type(blurred, dtype)

    @staticmethod
    def sobel_edges(image: np.ndarray, to_gray: bool = True) -> np.ndarray:
        """Edge magnitude using Sobel operator; returns grayscale unless told otherwise."""
        img_f, dtype, cs = Filters._to_float(image)
        gray = (
            0.299 * img_f[..., 0] + 0.587 * img_f[..., 1] + 0.114 * img_f[..., 2]
            if cs == Filters._RGB
            else img_f
        )
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        gx = Filters._convolve2d(gray, kx)
        gy = Filters._convolve2d(gray, ky)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        if to_gray or cs == Filters._GRAYSCALE:
            return Filters._restore_type(mag, dtype)
        edges_rgb = np.stack([mag] * 3, axis=-1)
        return Filters._restore_type(edges_rgb, dtype)

    @staticmethod
    def sharpen(image: np.ndarray, alpha: float = 1.0, radius: int = 2) -> np.ndarray:
        """Unsharp‑mask sharpening (Gaussian blur → subtract → add αlpha*mask)."""
        if alpha < 0:
            raise ValueError("alpha must be non‑negative")
        img_f, dtype, cs = Filters._to_float(image)
        kernel = Filters._gaussian_kernel(radius)
        if cs == Filters._GRAYSCALE:
            blurred = Filters._convolve2d(img_f, kernel)
        else:
            blurred = np.stack(
                [Filters._convolve2d(img_f[..., c], kernel) for c in range(3)],
                axis=-1,
            )
        mask = img_f - blurred
        sharpened = img_f + alpha * mask
        return Filters._restore_type(sharpened, dtype)

    # ---------------------------------------------------------------------
    # ──────────────────────────  ADJUSTMENTS  ────────────────────────────
    # ---------------------------------------------------------------------

    @staticmethod
    def adjust_brightness(image: np.ndarray, value: float) -> np.ndarray:
        """Add *value* to every pixel (positive = brighter, negative = darker)."""
        img_f, dtype, _ = Filters._to_float(image)
        return Filters._restore_type(img_f + value, dtype)

    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """Scale contrast by *factor* around mid‑grey 128."""
        img_f, dtype, _ = Filters._to_float(image)
        return Filters._restore_type((img_f - 128.0) * factor + 128.0, dtype)

    @staticmethod
    def adjust_saturation(image: np.ndarray, factor: float) -> np.ndarray:
        """Linear saturation change. 0 = grayscale, 1 = original, >1 = more color."""
        img_f, dtype, cs = Filters._to_float(image)
        if cs == Filters._GRAYSCALE:
            return Filters._restore_type(img_f, dtype)
        gray = 0.299 * img_f[..., 0] + 0.587 * img_f[..., 1] + 0.114 * img_f[..., 2]
        gray = gray[..., None]
        sat = gray + factor * (img_f - gray)
        return Filters._restore_type(sat, dtype)

    # ---------------------------------------------------------------------
    # ──────────────────────  PIPELINE (public)  ─────────────────────────--
    # ---------------------------------------------------------------------


    @staticmethod
    def apply_operations(image: np.ndarray, ops: List[dict]) -> np.ndarray:
        """Execute the *operations* array from the JSON config on *image*."""
        # ---------------- CONFIG‑list mode ----------------
        out = image
        for op in ops:
            out = Filters._apply_cfg_op(image, op)
        return out

    # Internal mapping helper ------------------------------------------------
    @staticmethod
    def _apply_cfg_op(img: np.ndarray, op: dict) -> np.ndarray:
        """Map a validated operation dict to the corresponding filter call."""
        t = op["type"]
        try:
            if t == "brightness":
                return Filters.adjust_brightness(img, op["value"])
            if t == "contrast":
                return Filters.adjust_contrast(img, op["value"])
            if t == "saturation":
                return Filters.adjust_saturation(img, op["value"])
            if t == "box":
                return Filters.box_blur(img, op["width"], op["height"])
            if t == "sharpen":
                return Filters.sharpen(img, op["alpha"])
            if t == "sobel":
                return Filters.sobel_edges(img)
        except KeyError as missing:
            raise ValueError(f"Operation '{t}' missing parameter {missing!s}") from None
        raise ValueError(f"Unsupported operation '{t}'")

    # ------------------------------------------------------------------
    # Convenience I/O / display utilities
    # ------------------------------------------------------------------

    @staticmethod
    def show(image: np.ndarray) -> None:
        """Display `image` in a matplotlib window (no return value)."""
        Filters._validate_image(image)
        img_disp = Filters._restore_type(image.astype(np.float32), np.uint8)
        if img_disp.ndim == 2:
            plt.imshow(img_disp, cmap="gray", vmin=0, vmax=255)
        else:
            plt.imshow(img_disp)
        plt.axis("off")
        plt.show()

    @staticmethod
    def save(image: np.ndarray, path: str) -> None:
        """Save `image` to *path* using matplotlib's imsave (supports PNG/JPG)."""
        Filters._validate_image(image)

        # ensure directory exists
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        img_out = Filters._restore_type(image.astype(np.float32), np.uint8)
        plt.imsave(path, img_out, cmap="gray" if img_out.ndim == 2 else None)
