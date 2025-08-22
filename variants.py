'''
variants.py
~~~~~~~~~~~
Custom image–frequency transforms for MMSegmentation pipelines.

Registered classes (use in config with dict(type=...))
-----------------------------------------------------
1. GaussianBlurVariant      ― Low‑pass (sigma)
2. UnsharpMaskVariant       ― High‑boost (sigma, amount)
3. FFTLowPassVariant        ― Low‑pass  (radius)
4. FFTHighPassVariant       ― High‑pass (radius)
5. GaussianNoiseVariant     ― Additive white Gaussian noise (std)

All classes expect **results['img']** from `LoadImageFromFile`:
    • H×W×C  BGR  np.uint8
and return the modified results dict.
'''

from __future__ import annotations

import cv2
import numpy as np
from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform  # mmcv>=2.0

# -----------------------------------------------------------------------------
# Helper: FFT circular mask
# -----------------------------------------------------------------------------

def _fft_filter(img: np.ndarray, radius: int, keep_low: bool) -> np.ndarray:
    """Apply circular low/high‑pass filter on each channel (BGR).

    Args:
        img (np.ndarray): uint8 BGR image (H, W, 3).
        radius (int): radius in pixels for the circular mask (in frequency domain).
        keep_low (bool): If True, keep low frequencies (LPF); else HPF.
    Returns:
        np.ndarray: uint8 BGR image after inverse FFT.
    """
    h, w = img.shape[:2]
    # Prepare meshgrid (centered at h//2, w//2)
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    dist2 = (y - cy) ** 2 + (x - cx) ** 2
    mask = dist2 <= radius ** 2  # True==low freq

    out = np.empty_like(img, dtype=np.float32)
    for c in range(3):  # BGR channels
        f = np.fft.fftshift(np.fft.fft2(img[:, :, c].astype(np.float32)))
        f_filtered = f * mask if keep_low else f * (~mask)
        img_back = np.fft.ifft2(np.fft.ifftshift(f_filtered)).real
        out[:, :, c] = img_back

    # Normalize back to uint8 range
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# -----------------------------------------------------------------------------
# Low‑pass: Gaussian blur
# -----------------------------------------------------------------------------

@TRANSFORMS.register_module()
class GaussianBlurVariant(BaseTransform):
    """Apply Gaussian blur (low‑pass) before Resize/Normalize."""

    def __init__(self, sigma: float = 1.5):
        self.sigma = float(sigma)

    def transform(self, results: dict) -> dict:
        img = results['img']

        # kernel size
        k = max(3, int(4 * self.sigma + 1))
        if k % 2 == 0:
            k += 1

        img = cv2.GaussianBlur(img, (k, k), self.sigma)
        results['img'] = img
        return results

    def __repr__(self) -> str:  # noqa: D401
        return f'{self.__class__.__name__}(sigma={self.sigma})'

# -----------------------------------------------------------------------------
# High‑pass: Unsharp mask (original + k*(original‑GaussianBlur))
# -----------------------------------------------------------------------------

@TRANSFORMS.register_module()
class UnsharpMaskVariant(BaseTransform):
    """Sharpen image via unsharp masking.

    Args:
        sigma (float): Gaussian σ used to compute the blurred image.
        amount (float): Scaling factor for high‑frequency details.
    """

    def __init__(self, sigma: float = 1.0, amount: float = 1.5):
        self.sigma = float(sigma)
        self.amount = float(amount)

    def transform(self, results: dict) -> dict:
        img = results['img']
        k = max(3, int(4 * self.sigma + 1))
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(img, (k, k), self.sigma)
        # sharpen = img + amount * (img - blurred)
        sharpen = cv2.addWeighted(img, 1 + self.amount, blurred, -self.amount, 0)
        results['img'] = np.clip(sharpen, 0, 255).astype(np.uint8)
        return results

    def __repr__(self) -> str:  # noqa: D401
        return f'{self.__class__.__name__}(sigma={self.sigma}, amount={self.amount})'

# -----------------------------------------------------------------------------
# FFT‑based Low‑pass / High‑pass
# -----------------------------------------------------------------------------

@TRANSFORMS.register_module()
class FFTLowPassVariant(BaseTransform):
    """Circular low‑pass filter in frequency domain."""

    def __init__(self, radius: int = 30):
        self.radius = int(radius)

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = _fft_filter(img, self.radius, keep_low=True)
        return results

    def __repr__(self) -> str:  # noqa: D401
        return f'{self.__class__.__name__}(radius={self.radius})'

@TRANSFORMS.register_module()
class FFTHighPassVariant(BaseTransform):
    """Circular high‑pass filter in frequency domain."""

    def __init__(self, radius: int = 30):
        self.radius = int(radius)

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = _fft_filter(img, self.radius, keep_low=False)
        return results

    def __repr__(self) -> str:  # noqa: D401
        return f'{self.__class__.__name__}(radius={self.radius})'

# -----------------------------------------------------------------------------
# Additive Gaussian Noise
# -----------------------------------------------------------------------------

@TRANSFORMS.register_module()
class GaussianNoiseVariant(BaseTransform):
    """Add additive white Gaussian noise to the image.

    Args:
        std (float): Standard deviation of the noise. Measured in 0‑255 scale.
                      e.g., std=10 adds noise ~ N(0, 10^2).
    """

    def __init__(self, std: float = 10.0):
        self.std = float(std)

    def transform(self, results: dict) -> dict:
        img = results['img'].astype(np.float32)
        noise = np.random.normal(loc=0.0, scale=self.std, size=img.shape).astype(np.float32)
        noised = img + noise
        results['img'] = np.clip(noised, 0, 255).astype(np.uint8)
        return results

    def __repr__(self) -> str:  # noqa: D401
        return f'{self.__class__.__name__}(std={self.std})'