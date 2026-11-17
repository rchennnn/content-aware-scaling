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

def get_neighbors(r, c, min_h, max_h, min_w, max_w):
    dirs = [(-1, 0), (-1, 1), (-1, -1), (0, 1), (0, -1), (1, 1), (1, 0), (1, -1)]
    neighbors = []
    for d in dirs:
        new_r = r + d[0]
        new_c = c + d[1]
        if new_r < min_h or new_r >= max_h:
            continue
        if new_c < min_w or new_c >= max_w:
            continue
        neighbors.append((new_r, new_c))
    return neighbors

def get_distance(my_vec, neighbor_vec): 
    bcast_diff = (my_vec - neighbor_vec)**2
    row_sums = bcast_diff.sum(axis=1)
    avg_dist = np.mean(row_sums)
    return avg_dist
    

DIRS = [(-1, 0), (-1, 1), (-1, -1), (0, 1), (0, -1), (1, 1), (1, 0), (1, -1)]

def get_distance_matrix(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    _, height, width = image.shape
    total: npt.NDArray[np.float32] = np.zeros((height, width), dtype=np.float32)
    count: npt.NDArray[np.float32] = np.zeros((height, width), dtype=np.float32)
    for dr, dc in DIRS:
        # pixels whose neighbor (r+dr, c+dc) is inside the image
        r0, r1 = max(0, -dr), height - max(0, dr)
        c0, c1 = max(0, -dc), width - max(0, dc)
        diff = image[:, r0:r1, c0:c1] - image[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        total[r0:r1, c0:c1] += (diff ** 2).sum(axis=0)
        count[r0:r1, c0:c1] += 1
    return total / count


def find_min_path(distance_matrix: npt.NDArray[np.float32]) -> tuple[list[int], float]:
    """Top-to-bottom path (one column per row, each step moves col by -1/0/+1)
    with the least average distance. Returns (cols, avg) where cols[r] is the
    column at row r. Every path has length H, so min average == min sum.

    DP: cost[r, c] = distance_matrix[r, c] + min(cost[r-1, c-1 : c+2])
    """
    path, total = _min_path_dp(np.ascontiguousarray(distance_matrix, dtype=np.float64))
    cols: list[int] = path.tolist()
    avg: float = float(total / distance_matrix.shape[0])
    return cols, avg


@njit(cache=True)
def _min_path_dp(dist: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.int64], float]:
    """Compiled DP + backtrack. Returns (cols, total_cost).
    Ties prefer upper-left, then up, then upper-right; last row takes the
    leftmost minimum."""
    height, width = dist.shape
    cost: npt.NDArray[np.float64] = np.empty((height, width), dtype=np.float64)
    parent: npt.NDArray[np.int64] = np.empty((height, width), dtype=np.int64)
    for c in range(width):
        cost[0, c] = dist[0, c]
    for r in range(1, height):
        for c in range(width):
            best_c: int = c
            best: float = cost[r - 1, c]
            if c > 0 and cost[r - 1, c - 1] <= best:   # left wins ties
                best_c = c - 1
                best = cost[r - 1, c - 1]
            if c < width - 1 and cost[r - 1, c + 1] < best:
                best_c = c + 1
                best = cost[r - 1, c + 1]
            cost[r, c] = dist[r, c] + best
            parent[r, c] = best_c

    end: int = 0
    for c in range(1, width):
        if cost[height - 1, c] < cost[height - 1, end]:
            end = c
    path: npt.NDArray[np.int64] = np.empty(height, dtype=np.int64)
    path[height - 1] = end
    for r in range(height - 1, 0, -1):
        path[r - 1] = parent[r, path[r]]
    return path, cost[height - 1, end]


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


def compute_distance_at(
    image: npt.NDArray[np.float32],
    rows: npt.NDArray[np.int64],
    cols: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Same per-pixel value as get_distance_matrix, but only at the given
    (rows[i], cols[i]) coordinates."""
    _, height, width = image.shape
    total: npt.NDArray[np.float32] = np.zeros(rows.shape, dtype=np.float32)
    count: npt.NDArray[np.float32] = np.zeros(rows.shape, dtype=np.float32)
    for dr, dc in DIRS:
        nr: npt.NDArray[np.int64] = rows + dr
        nc: npt.NDArray[np.int64] = cols + dc
        valid: npt.NDArray[np.bool_] = (nr >= 0) & (nr < height) & (nc >= 0) & (nc < width)
        diff: npt.NDArray[np.float32] = image[:, rows[valid], cols[valid]] - image[:, nr[valid], nc[valid]]
        total[valid] += (diff ** 2).sum(axis=0)
        count[valid] += 1
    return total / count


def carve(
    image: npt.NDArray[np.float32],
    threshold: float,
    max_paths: int | None = None,
    verbose: bool = True,
    band: int = 3,
) -> tuple[npt.NDArray[np.float32], list[list[int]], list[float]]:
    """Repeatedly find the least-average-distance top-to-bottom path, remove
    it, and stitch the image back together. Stops when the best path's average
    distance is >= threshold (or after max_paths removals, or width hits 1).

    Returns (carved_image, removed_paths, removed_avgs). Each removed path's
    columns are in the coordinates of the image at the time it was removed.
    """
    removed_paths: list[list[int]] = []
    removed_avgs: list[float] = []
    # total is an upper bound; the threshold usually stops it earlier
    limit: int = image.shape[2] - 1 if max_paths is None else min(max_paths, image.shape[2] - 1)
    pbar: tqdm = tqdm(total=limit, desc="carving", unit="path", disable=not verbose)
    height: int = image.shape[1]
    offsets: npt.NDArray[np.int64] = np.arange(-band, band + 1)
    band_rows: npt.NDArray[np.int64] = np.repeat(np.arange(height), len(offsets))
    dist: npt.NDArray[np.float32] = get_distance_matrix(image)  # full compute once
    while image.shape[2] > 1 and (max_paths is None or len(removed_paths) < max_paths):
        cols, avg = find_min_path(dist)
        if avg >= threshold:
            break
        image = remove_path(image, cols)
        dist = remove_path(dist, cols)
        # only pixels within +/- band columns of the removed path changed neighbors
        band_cols: npt.NDArray[np.int64] = (
            np.asarray(cols)[:, None] + offsets
        ).clip(0, image.shape[2] - 1).ravel()
        dist[band_rows, band_cols] = compute_distance_at(image, band_rows, band_cols)
        removed_paths.append(cols)
        removed_avgs.append(avg)
        pbar.update(1)
        pbar.set_postfix(width=image.shape[2], avg=f"{avg:.3g}")
    pbar.close()
    return image, removed_paths, removed_avgs


def carve_rows(
    image: npt.NDArray[np.float32],
    threshold: float,
    max_paths: int | None = None,
    verbose: bool = True,
    band: int = 3,
) -> tuple[npt.NDArray[np.float32], list[list[int]], list[float]]:
    """Same as carve, but removes left-to-right paths (one row per column,
    each step moves row by -1/0/+1), shrinking the height.

    Implemented by transposing H and W, carving, and transposing back.
    Returns (carved_image, removed_paths, removed_avgs) where removed_paths[k][c]
    is the row removed at column c for the k-th path.
    """
    transposed: npt.NDArray[np.float32] = image.transpose(0, 2, 1)  # (3, W, H)
    carved_t, removed_paths, removed_avgs = carve(transposed, threshold, max_paths, verbose, band)
    carved: npt.NDArray[np.float32] = carved_t.transpose(0, 2, 1)   # back to (3, H', W)
    return carved, removed_paths, removed_avgs


def save_image(image: npt.NDArray[np.float32], path: str) -> None:
    """Save a (3, H, W) float image (values in [0, 1)) as a PNG."""
    hwc: npt.NDArray[np.uint8] = (image.transpose(1, 2, 0) * 256).clip(0, 255).astype(np.uint8)
    Image.fromarray(hwc).save(path)


if __name__ == "__main__":
    arr: npt.NDArray[np.float32] = load_image("IMG_6081.png")
    THRESHOLD=1e-3
    print(f"original: {arr.shape}")
    carved, removed, avgs = carve(arr, threshold=THRESHOLD)
    print(f"removed {len(removed)} column paths -> {carved.shape}")
    if avgs:
        print(f"avg distance of removed paths: first {avgs[0]:.3g}, last {avgs[-1]:.3g}")
    carved, removed, avgs = carve_rows(carved, threshold=THRESHOLD)
    print(f"removed {len(removed)} row paths -> {carved.shape}")
    if avgs:
        print(f"avg distance of removed paths: first {avgs[0]:.3g}, last {avgs[-1]:.3g}")
    save_image(carved, "carved.png")
    print("saved carved.png")
