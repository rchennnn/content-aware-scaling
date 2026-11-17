import math

import numpy as np
import numpy.typing as npt
from PIL import Image
from numba import njit
from tqdm import tqdm


def load_image(path: str) -> npt.NDArray[np.float32]:
    """Load an image as a (3, H, W) uint8 array (RGB channels first)."""
    img: Image.Image = Image.open(path).convert("RGB")  # drops alpha / expands grayscale
    arr: npt.NDArray[np.uint8] = np.asarray(img)       # (H, W, 3)
    arr: npt.NDArray[np.float32] = arr.astype(np.float32) / 256
    return arr.transpose(2, 0, 1)                       # (3, H, W)

def luminance(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """(3, H, W) RGB -> (H, W) perceived brightness."""
    return (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]).astype(np.float32)


@njit(cache=True)
def _local_mad(values: npt.NDArray[np.float32], radius: int) -> npt.NDArray[np.float32]:
    """Median absolute deviation over a (2*radius+1)^2 window around each
    pixel, with the window clipped at the borders."""
    height, width = values.shape
    out: npt.NDArray[np.float32] = np.empty((height, width), dtype=np.float32)
    buf: npt.NDArray[np.float32] = np.empty((2 * radius + 1) ** 2, dtype=np.float32)
    for r in range(height):
        r0, r1 = max(0, r - radius), min(height, r + radius + 1)
        for c in range(width):
            c0, c1 = max(0, c - radius), min(width, c + radius + 1)
            n: int = 0
            for i in range(r0, r1):
                for j in range(c0, c1):
                    buf[n] = values[i, j]
                    n += 1
            window = buf[:n]
            m = np.median(window)
            out[r, c] = np.median(np.abs(window - m))
    return out


def variability_map(
    image: npt.NDArray[np.float32], radius: int = 7, stride: int = 4
) -> npt.NDArray[np.float32]:
    """Robust local variability of each pixel's neighbourhood, as a (H, W) map
    on the same scale as a standard deviation.

    Uses horizontal neighbour differences of the brightness rather than the
    brightness itself, so a smooth gradient (constant differences) scores ~0,
    texture (differences that jump around) scores high, and a single sharp
    edge is an outlier the median ignores.

    For speed the differences are sampled every `stride` pixels, the MAD is
    taken over a (2*radius+1)^2 window on that grid (so it spans about
    (2*radius+1)*stride original pixels), and the result is scaled back up.
    """
    _, height, width = image.shape
    gray: npt.NDArray[np.float32] = luminance(image)
    diffs: npt.NDArray[np.float32] = np.empty_like(gray)
    diffs[:, :-1] = gray[:, 1:] - gray[:, :-1]
    diffs[:, -1] = diffs[:, -2]
    small: npt.NDArray[np.float32] = _local_mad(np.ascontiguousarray(diffs[::stride, ::stride]), radius)
    full: npt.NDArray[np.float32] = np.repeat(np.repeat(small, stride, axis=0), stride, axis=1)
    return 1.4826 * full[:height, :width]


def find_min_path(
    image: npt.NDArray[np.float32],
    inv_scale: npt.NDArray[np.float32],
    step_penalty: float = 0.0,
) -> tuple[list[int], float]:
    """Top-to-bottom path (one column per row, each step moves col by -1/0/+1)
    whose removal leaves the least visible joins. Returns (cols, avg) where
    cols[r] is the column at row r and avg is the path's mean cost per row.

    Costs use forward energy: instead of asking how busy the removed pixels
    were, we ask how different the pixels are that become neighbours once
    they are gone. Removing (r, c) makes (r, c-1) and (r, c+1) adjacent, and
    arriving diagonally also makes (r-1, c) adjacent to (r, c-1) or (r, c+1).

    Each such join is scored as z = colour mismatch / local variability
    (inv_scale holds 1 / variability), and only the part beyond natural
    variation counts: cost = max(0, z - 1). A join that looks like the
    texture already around it is free, so paths can run through rippling
    water or smooth sky at no cost, while sharp edges stay expensive.

    step_penalty optionally adds a fixed charge to every diagonal step. The
    returned avg excludes it, so the threshold means the same with or without.
    """
    path, total = _min_path_forward(
        np.ascontiguousarray(image), np.ascontiguousarray(inv_scale), step_penalty
    )
    return path.tolist(), float(total / image.shape[1])


@njit(cache=True, fastmath=True, error_model="numpy")
def _join_cost(
    image: npt.NDArray[np.float32], inv: float, r1: int, c1: int, r2: int, c2: int
) -> float:
    """max(0, z - 1) for the join between pixels (r1, c1) and (r2, c2), where z
    is their RMS colour difference across the three channels times inv
    (1 / variability)."""
    d0 = image[0, r1, c1] - image[0, r2, c2]
    d1 = image[1, r1, c1] - image[1, r2, c2]
    d2 = image[2, r1, c1] - image[2, r2, c2]
    z = np.sqrt((d0 * d0 + d1 * d1 + d2 * d2) * np.float32(1.0 / 3.0)) * inv
    return z - 1.0 if z > 1.0 else 0.0


@njit(cache=True, fastmath=True, error_model="numpy")
def _min_path_forward(
    image: npt.NDArray[np.float32], inv_scale: npt.NDArray[np.float32], step_penalty: float
) -> tuple[npt.NDArray[np.int64], float]:
    """Compiled forward-energy DP + backtrack. Returns (cols, total_cost), where
    total_cost sums the join costs along the path without step penalties.
    Ties prefer upper-left, then up, then upper-right; last row takes the
    leftmost minimum. At the borders, a missing neighbour falls back to the
    pixel itself, so removing an edge pixel costs its ordinary neighbour
    difference rather than nothing."""
    _, height, width = image.shape
    cost: npt.NDArray[np.float64] = np.empty((height, width), dtype=np.float64)
    parent: npt.NDArray[np.int64] = np.empty((height, width), dtype=np.int64)
    step: npt.NDArray[np.float64] = np.empty((height, width), dtype=np.float64)
    for r in range(height):
        for c in range(width):
            inv: float = inv_scale[r, c]
            left: int = c - 1 if c > 0 else c
            right: int = c + 1 if c < width - 1 else c
            cu: float = _join_cost(image, inv, r, left, r, right)
            if r == 0:
                cost[r, c] = cu
                step[r, c] = cu
                continue
            best_c: int = c
            best_step: float = cu
            best: float = cost[r - 1, c] + cu
            if c > 0:
                cl: float = cu + _join_cost(image, inv, r - 1, c, r, c - 1)
                v: float = cost[r - 1, c - 1] + cl + step_penalty
                if v <= best:   # left wins ties
                    best_c, best, best_step = c - 1, v, cl
            if c < width - 1:
                cr: float = cu + _join_cost(image, inv, r - 1, c, r, c + 1)
                v = cost[r - 1, c + 1] + cr + step_penalty
                if v < best:
                    best_c, best, best_step = c + 1, v, cr
            cost[r, c] = best
            parent[r, c] = best_c
            step[r, c] = best_step

    end: int = 0
    for c in range(1, width):
        if cost[height - 1, c] < cost[height - 1, end]:
            end = c
    path: npt.NDArray[np.int64] = np.empty(height, dtype=np.int64)
    path[height - 1] = end
    for r in range(height - 1, 0, -1):
        path[r - 1] = parent[r, path[r]]
    total: float = 0.0
    for r in range(height):
        total += step[r, path[r]]
    return path, total


def find_min_paths_from_each_start(
    distance_matrix: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """For every starting column in row 0, the least-average-distance path to
    the bottom row (each step moves col by -1/0/+1).

    Returns (paths, avgs):
      paths: (W, H) — paths[c0, r] is the column at row r for the path starting at c0
      avgs:  (W,)   — avgs[c0] is that path's average distance

    Runs the DP bottom-up so cost[r, c] = cheapest cost from (r, c) to the
    bottom row; then cost[0, c0] answers every start in one O(H*W) pass.
    """
    height, width = distance_matrix.shape
    cost: npt.NDArray[np.float64] = np.zeros((height, width), dtype=np.float64)
    child: npt.NDArray[np.int64] = np.zeros((height, width), dtype=np.int64)
    cost[-1] = distance_matrix[-1]

    cols_idx: npt.NDArray[np.int64] = np.arange(width)
    for r in range(height - 2, -1, -1):
        nxt: npt.NDArray[np.float64] = cost[r + 1]
        # candidates at lower-left, down, lower-right; out-of-bounds = inf
        left: npt.NDArray[np.float64] = np.concatenate(([np.inf], nxt[:-1]))
        right: npt.NDArray[np.float64] = np.concatenate((nxt[1:], [np.inf]))
        options: npt.NDArray[np.float64] = np.stack([left, nxt, right])  # (3, W)
        choice: npt.NDArray[np.int64] = options.argmin(axis=0)          # 0,1,2
        cost[r] = distance_matrix[r] + options[choice, cols_idx]
        child[r] = cols_idx + choice - 1

    # trace all W paths forward at once
    paths: npt.NDArray[np.int64] = np.zeros((width, height), dtype=np.int64)
    paths[:, 0] = cols_idx
    for r in range(height - 1):
        paths[:, r + 1] = child[r, paths[:, r]]
    avgs: npt.NDArray[np.float64] = cost[0] / height
    return paths, avgs


def remove_path(image: npt.NDArray[np.float32], cols: list[int]) -> npt.NDArray[np.float32]:
    """Delete pixel (r, cols[r]) from every row and shift the rest left.
    Works on (3, H, W) -> (3, H, W-1) and on (H, W) -> (H, W-1)."""
    cols_arr: npt.NDArray[np.int64] = np.asarray(cols, dtype=np.int64)
    if image.ndim == 2:
        return _remove_path_kernel(image[None], cols_arr)[0]
    return _remove_path_kernel(image, cols_arr)


@njit(cache=True)
def _remove_path_kernel(
    image: npt.NDArray[np.float32], cols: npt.NDArray[np.int64]
) -> npt.NDArray[np.float32]:
    """Compiled row-by-row copy that skips pixel cols[r]. (C, H, W) -> (C, H, W-1).
    Output is always C-contiguous, even if the input is a transposed view."""
    channels, height, width = image.shape
    out: npt.NDArray[np.float32] = np.empty((channels, height, width - 1), dtype=image.dtype)
    for ch in range(channels):
        for r in range(height):
            c0: int = cols[r]
            for c in range(c0):
                out[ch, r, c] = image[ch, r, c]
            for c in range(c0 + 1, width):
                out[ch, r, c - 1] = image[ch, r, c]
    return out


# Expected cost of a single join between two pixels of ordinary texture.
# If a join's z (colour difference / local variability) behaves like the
# absolute value of a standard normal variable, then
#     E[max(0, |Z| - 1)] = 2 * (phi(1) - (1 - Phi(1))) ~= 0.167,
# where phi and Phi are the standard normal density and CDF. A path whose
# average cost per row is at or below this leaves joins no more unusual than
# the texture already around them, so removing it is effectively free. This
# number comes from the cost's definition, not from tuning.
FREE_COST: float = 2.0 * (
    math.exp(-0.5) / math.sqrt(2.0 * math.pi) - 0.5 * math.erfc(1.0 / math.sqrt(2.0))
)


def find_elbow(costs: list[float]) -> int:
    """Given the cost of each removed path in order, return how many of them
    to keep: the point where removals stop being cheap and start to climb.

    Works on the cumulative cost, which rises slowly while the carver eats
    filler and steeply once only content is left. After scaling both axes to
    [0, 1], the elbow is the point that sits furthest below the straight line
    from the first point to the last. Equivalently, it is where a single
    path's cost first overtakes the average cost of all the paths tried, so
    it needs no threshold, only the curve's own shape. Using the cumulative
    cost rather than the raw per-path costs makes it robust to noise.
    """
    n: int = len(costs)
    if n == 0:
        return 0
    cumulative: npt.NDArray[np.float64] = np.cumsum(np.asarray(costs, dtype=np.float64))
    if cumulative[-1] <= 0:
        return n  # every path was free, so keep them all
    x: npt.NDArray[np.float64] = np.arange(1, n + 1) / n
    y: npt.NDArray[np.float64] = cumulative / cumulative[-1]
    return int(np.argmax(x - y)) + 1


def carve(
    image: npt.NDArray[np.float32],
    stop: str | float = "free",
    max_paths: int | None = None,
    verbose: bool = True,
    mad_radius: int = 7,
    mad_stride: int = 4,
    mad_floor: float = 1 / 256,
    step_penalty: float = 0.0,
    min_keep: float = 0.25,
) -> tuple[npt.NDArray[np.float32], list[list[int]], list[float]]:
    """Repeatedly find the top-to-bottom path whose removal leaves the least
    visible joins (see find_min_path), remove it, and stitch the image back
    together. `stop` decides when to finish:

    "free" (default): stop as soon as the cheapest remaining path is no
        longer free, meaning its average cost per row exceeds FREE_COST, the
        cost of an ordinary join inside natural texture. Everything nobody
        would notice is removed, and nothing more.
    "elbow": keep removing paths until only min_keep of the width is left,
        use find_elbow on the recorded costs to pick the point where removals
        start to get expensive, and rewind to it by replaying just those
        removals on the original image. min_keep only limits how far the
        search goes; it matters only if the elbow is never reached.
    a number: stop as soon as the best path's average cost reaches it.

    In every mode the carver also stops after max_paths removals, or when the
    width would drop to 1.

    Each join is measured against sigma, the robust local variability from
    variability_map, floored at mad_floor (one 8-bit brightness level by
    default) so perfectly flat regions never divide by zero. step_penalty is
    passed to find_min_path to discourage diagonal steps.

    Returns (carved_image, removed_paths, removed_avgs). Each removed path's
    columns are in the coordinates of the image at the time it was removed.
    """
    original: npt.NDArray[np.float32] = image
    removed_paths: list[list[int]] = []
    removed_avgs: list[float] = []
    if stop == "free":
        threshold: float | None = FREE_COST
    elif stop == "elbow":
        threshold = None
    elif isinstance(stop, (int, float)):
        threshold = float(stop)
    else:
        raise ValueError(f'stop must be "free", "elbow" or a number, not {stop!r}')
    limit: int = image.shape[2] - 1
    if threshold is None:
        limit = min(limit, image.shape[2] - max(1, int(round(image.shape[2] * min_keep))))
    if max_paths is not None:
        limit = min(limit, max_paths)
    pbar: tqdm = tqdm(total=limit, desc="carving", unit="path", disable=not verbose)
    # variability is a property of the region, so compute it once and carve
    # it alongside the image instead of recomputing it every iteration
    sigma: npt.NDArray[np.float32] = variability_map(image, mad_radius, mad_stride)
    inv_scale: npt.NDArray[np.float32] = (1.0 / np.sqrt(sigma ** 2 + mad_floor ** 2)).astype(np.float32)
    while len(removed_paths) < limit:
        cols, avg = find_min_path(image, inv_scale, step_penalty)
        if threshold is not None and avg > threshold:
            break
        image = remove_path(image, cols)
        inv_scale = remove_path(inv_scale, cols)
        removed_paths.append(cols)
        removed_avgs.append(avg)
        pbar.update(1)
        pbar.set_postfix(width=image.shape[2], avg=f"{avg:.3g}")
    pbar.close()

    if threshold is None:
        keep: int = find_elbow(removed_avgs)
        if verbose:
            print(f"elbow at path {keep} of {len(removed_paths)}; rewinding")
        removed_paths, removed_avgs = removed_paths[:keep], removed_avgs[:keep]
        # replaying recorded removals needs no path search, so it is quick
        image = np.ascontiguousarray(original)
        for cols in removed_paths:
            image = remove_path(image, cols)
    return image, removed_paths, removed_avgs


def carve_rows(
    image: npt.NDArray[np.float32],
    stop: str | float = "free",
    max_paths: int | None = None,
    verbose: bool = True,
    mad_radius: int = 7,
    mad_stride: int = 4,
    mad_floor: float = 1 / 256,
    step_penalty: float = 0.0,
    min_keep: float = 0.25,
) -> tuple[npt.NDArray[np.float32], list[list[int]], list[float]]:
    """Same as carve, but removes left-to-right paths (one row per column,
    each step moves row by -1/0/+1), shrinking the height.

    Implemented by transposing H and W, carving, and transposing back. Because
    of the transpose, the variability map is built from vertical differences.
    Returns (carved_image, removed_paths, removed_avgs) where removed_paths[k][c]
    is the row removed at column c for the k-th path.
    """
    transposed: npt.NDArray[np.float32] = image.transpose(0, 2, 1)  # (3, W, H)
    carved_t, removed_paths, removed_avgs = carve(
        transposed, stop, max_paths, verbose, mad_radius, mad_stride, mad_floor,
        step_penalty, min_keep,
    )
    carved: npt.NDArray[np.float32] = carved_t.transpose(0, 2, 1)   # back to (3, H', W)
    return carved, removed_paths, removed_avgs


def save_image(image: npt.NDArray[np.float32], path: str) -> None:
    """Save a (3, H, W) float image (values in [0, 1)) as a PNG."""
    hwc: npt.NDArray[np.uint8] = (image.transpose(1, 2, 0) * 256).clip(0, 255).astype(np.uint8)
    Image.fromarray(hwc).save(path)


def removed_mask(
    shape: tuple[int, int],
    col_paths: list[list[int]],
    row_paths: list[list[int]],
) -> npt.NDArray[np.bool_]:
    """Map the paths removed by carve (col_paths) and then carve_rows
    (row_paths) back onto the original image. Returns an (H, W) boolean mask
    that is True for every original pixel that was carved away.

    Each path is stored in the coordinates of the image at the moment it was
    removed, so we replay the removals on two index maps that remember each
    surviving pixel's original row and column.
    """
    height, width = shape
    orig_r: npt.NDArray[np.int32] = np.repeat(np.arange(height, dtype=np.int32)[:, None], width, axis=1)
    orig_c: npt.NDArray[np.int32] = np.repeat(np.arange(width, dtype=np.int32)[None, :], height, axis=0)
    mask: npt.NDArray[np.bool_] = np.zeros(shape, dtype=np.bool_)
    for cols in col_paths:
        rows_idx: npt.NDArray[np.int64] = np.arange(len(cols))
        mask[orig_r[rows_idx, cols], orig_c[rows_idx, cols]] = True
        orig_r, orig_c = remove_path(orig_r, cols), remove_path(orig_c, cols)
    # carve_rows works on the transposed image, so replay on transposed maps
    orig_r, orig_c = orig_r.T, orig_c.T
    for rows in row_paths:
        cols_idx: npt.NDArray[np.int64] = np.arange(len(rows))
        mask[orig_r[cols_idx, rows], orig_c[cols_idx, rows]] = True
        orig_r, orig_c = remove_path(orig_r, rows), remove_path(orig_c, rows)
    return mask


def grey_out(image: npt.NDArray[np.float32], mask: npt.NDArray[np.bool_], grey: float = 0.5) -> npt.NDArray[np.float32]:
    """Return a copy of a (3, H, W) image with the masked pixels set to flat grey."""
    out: npt.NDArray[np.float32] = image.copy()
    out[:, mask] = grey
    return out


if __name__ == "__main__":
    arr: npt.NDArray[np.float32] = load_image("IMG_6081.png")
    print(f"original: {arr.shape}")
    # each pass removes paths until the cheapest one is no longer free
    carved, col_removed, avgs = carve(arr, stop=0.05)
    print(f"removed {len(col_removed)} column paths -> {carved.shape}")
    if avgs:
        print(f"avg cost of removed paths: first {avgs[0]:.3g}, last {avgs[-1]:.3g}")
    carved, row_removed, avgs = carve_rows(carved, stop=0.05)
    print(f"removed {len(row_removed)} row paths -> {carved.shape}")
    if avgs:
        print(f"avg cost of removed paths: first {avgs[0]:.3g}, last {avgs[-1]:.3g}")
    save_image(carved, "carved.png")
    print("saved carved.png")
    mask = removed_mask(arr.shape[1:], col_removed, row_removed)
    print(f"{int(mask.sum())} pixels removed ({mask.mean():.1%} of the original)")
    save_image(grey_out(arr, mask), "removed.png")
    print("saved removed.png")
