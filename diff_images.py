import numpy as np
import numpy.typing as npt
from PIL import Image


def bottom_aligned_diff(original_path: str, carved_path: str) -> npt.NDArray[np.float32]:
    """Align the carved image with the bottom (and left edge) of the original,
    take the per-pixel absolute colour difference averaged over RGB, and
    rescale it to [0, 1]. Returns a (H', W') grayscale array, where H' x W' is
    the carved image's size.

    If carving had only trimmed rows off the top, the bottom-aligned images
    would match exactly and the result would be black everywhere.
    """
    original: npt.NDArray[np.float32] = np.asarray(Image.open(original_path).convert("RGB"), dtype=np.float32)
    carved: npt.NDArray[np.float32] = np.asarray(Image.open(carved_path).convert("RGB"), dtype=np.float32)
    height, width, _ = carved.shape
    aligned: npt.NDArray[np.float32] = original[-height:, :width]
    diff: npt.NDArray[np.float32] = np.abs(aligned - carved).mean(axis=2)
    lo, hi = float(diff.min()), float(diff.max())
    return (diff - lo) / (hi - lo) if hi > lo else np.zeros_like(diff)


def save_gray(image: npt.NDArray[np.float32], path: str) -> None:
    """Save a (H, W) array with values in [0, 1] as an 8-bit grayscale PNG."""
    Image.fromarray((image * 255).round().astype(np.uint8), mode="L").save(path)


if __name__ == "__main__":
    diff = bottom_aligned_diff("IMG_6081.png", "carved.png")
    save_gray(diff, "diff.png")
    print(f"saved diff.png {diff.shape}, mean {diff.mean():.3f}")
