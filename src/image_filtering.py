"""Image filtering functions for Student 5 (2D Fourier & Image Filtering).

This module contains reusable functions for the Streamlit app and for
Student 6's image-processing work. It does not depend on Google Colab.
"""

import cv2
import numpy as np


def load_image(image_path):
    """Load a color image from a file path (OpenCV BGR format)."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def convert_to_grayscale(image):
    """Convert a BGR color image to a single-channel grayscale image."""
    if image is None:
        raise ValueError("Input image is empty.")
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Expected a grayscale image or a 3-channel BGR image.")


def compute_fourier_transform(gray_image):
    """Return the shifted 2D FFT of a grayscale image."""
    if gray_image is None or gray_image.ndim != 2:
        raise ValueError("A 2D grayscale image is required.")
    transform = np.fft.fft2(gray_image)
    shifted_transform = np.fft.fftshift(transform)
    return shifted_transform


def get_magnitude_spectrum(shifted_transform):
    """Return the log-scaled magnitude spectrum for display."""
    magnitude = np.abs(shifted_transform)
    return np.log1p(magnitude)


def create_circular_masks(image_shape, cutoff=30):
    """Create ideal circular low-pass and high-pass masks.

    The masks are centered to match a Fourier transform after fftshift.
    """
    rows, cols = image_shape[:2]
    center_row, center_col = rows // 2, cols // 2

    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)

    low_pass_mask = (distance <= cutoff).astype(np.uint8)
    high_pass_mask = 1 - low_pass_mask
    return low_pass_mask, high_pass_mask


def apply_frequency_mask(shifted_transform, mask):
    """Apply a frequency mask and return the real-valued inverse FFT result."""
    if shifted_transform.shape != mask.shape:
        raise ValueError("The mask and Fourier transform must have the same shape.")

    filtered_shifted = shifted_transform * mask
    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_shifted))
    return np.real(filtered_image)


def process_image(gray_image, cutoff=30):
    """Run Student 5's complete low-pass and high-pass filtering pipeline."""
    shifted_transform = compute_fourier_transform(gray_image)
    spectrum = get_magnitude_spectrum(shifted_transform)

    low_mask, high_mask = create_circular_masks(gray_image.shape, cutoff)

    low_result = apply_frequency_mask(shifted_transform, low_mask)
    high_result = apply_frequency_mask(shifted_transform, high_mask)

    # Normalize only the high-pass copy used for display.
    high_display = cv2.normalize(
        high_result, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    return {
        "fourier_shifted": shifted_transform,
        "magnitude_spectrum": spectrum,
        "low_pass_mask": low_mask,
        "high_pass_mask": high_mask,
        "low_pass_image": low_result,
        "high_pass_image": high_result,
        "high_pass_display": high_display,
    }
