import argparse
import os
import time

import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import rescale
from skimage.filters import sobel


# 1. Splitting the plate into channels
def split_channels(im):
    # split the stacked BGR glass-plate into three equal thirds.
    # returns (b, g, r) as float images in [0, 1].
    im = sk.img_as_float(im)
    h = im.shape[0] // 3
    b = im[0:h]
    g = im[h:2 * h]
    r = im[2 * h:3 * h]
    return b, g, r


# 2. Similarity metrics
def crop_border(im, frac=0.1):
    # Ignore a border strip when scoring alignment
    h, w = im.shape[:2]
    dy, dx = int(h * frac), int(w * frac)
    return im[dy:h - dy, dx:w - dx]


def ncc_score(a, b):
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.sum(a * b) / (na * nb))


def l2_score(a, b):
    return -float(np.sqrt(np.sum((a - b) ** 2)))


def score(a, b, metric):
    return ncc_score(a, b) if metric == "ncc" else l2_score(a, b)


# 3. Feature representations
def to_feature(im, features):
    # Convert channel image into whatever representation to be scored on

    # 'raw'  : plain pixel intensities. Works for most plates
    # 'edge' : gradient magnitude (Sobel)
    if features == "edge":
        return sobel(im)
    return im


# 4. Single-scale exhaustive search
def align_single_scale(moving, fixed, window=15, metric="ncc", features="raw"):
    # Try every integer (dy, dx) in [-window, window]^2, return best
    fixed_c = crop_border(to_feature(fixed, features))
    best_score = -np.inf
    best_shift = (0, 0)
    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            shifted = np.roll(np.roll(moving, dy, axis=0), dx, axis=1)
            shifted_f = crop_border(to_feature(shifted, features))
            s = score(shifted_f, fixed_c, metric)
            if s > best_score:
                best_score = s
                best_shift = (dy, dx)
    return best_shift


# 5. Coarse-to-fine pyramid alignment
def align_pyramid(moving, fixed, metric="ncc", features="raw",
                   coarse_window=15, refine_window=2, min_size=100):
    # Recursively downsample by 2x until the image is <= min_size on its longer side, 
    # align at that coarse scale with a wide exhaustive search, then double the estimate and refine at each finer level.
    h, w = fixed.shape[:2]
    if max(h, w) <= min_size:
        return align_single_scale(moving, fixed, window=coarse_window,
                                   metric=metric, features=features)

    # build the next coarser level
    moving_small = rescale(moving, 0.5, anti_aliasing=True, channel_axis=None)
    fixed_small = rescale(fixed, 0.5, anti_aliasing=True, channel_axis=None)

    dy, dx = align_pyramid(moving_small, fixed_small, metric=metric,
                            features=features, coarse_window=coarse_window,
                            refine_window=refine_window, min_size=min_size)
    dy, dx = 2 * dy, 2 * dx

    # refine at this resolution around the coarse estimate with a small window
    moving_shifted = np.roll(np.roll(moving, dy, axis=0), dx, axis=1)
    refine_dy, refine_dx = align_single_scale(
        moving_shifted, fixed, window=refine_window, metric=metric,
        features=features
    )
    return dy + refine_dy, dx + refine_dx


def align(moving, fixed, metric="ncc", features="raw",
          window=15, refine_window=2, min_size=400):
    # single-scale search for small images, pyramid for big ones.
    h, w = fixed.shape[:2]
    if max(h, w) <= min_size:
        return align_single_scale(moving, fixed, window=window,
                                   metric=metric, features=features)
    return align_pyramid(moving, fixed, metric=metric, features=features,
                          coarse_window=window, refine_window=refine_window,
                          min_size=min_size)


def apply_shift(im, shift):
    dy, dx = shift
    return np.roll(np.roll(im, dy, axis=0), dx, axis=1)


# 6. Bells & whistles
def auto_crop(im, max_border=0.12):
    # Trim the colored fringe / torn edges left after channel alignment.
    h, w = im.shape[:2]
    max_dy = int(h * max_border)
    max_dx = int(w * max_border)

    gray = im.mean(axis=2)
    col_var = gray.std(axis=0)
    row_var = gray.std(axis=1)

    def find_cut(profile, max_cut):
        med = np.median(profile)
        thresh = med * 0.4
        cut = 0
        for i in range(max_cut):
            if profile[i] < thresh:
                cut = i + 1
            else:
                break
        return cut

    top = find_cut(row_var, max_dy)
    bottom = find_cut(row_var[::-1], max_dy)
    left = find_cut(col_var, max_dx)
    right = find_cut(col_var[::-1], max_dx)

    # always trim a small fixed margin
    fixed_margin_y = max(int(h * 0.02), 2)
    fixed_margin_x = max(int(w * 0.02), 2)
    top = max(top, fixed_margin_y)
    bottom = max(bottom, fixed_margin_y)
    left = max(left, fixed_margin_x)
    right = max(right, fixed_margin_x)

    return im[top:h - bottom, left:w - right]


def auto_contrast(im, low_pct=1.0, high_pct=99.0):
    # Simple per-channel contrast stretch: clip the low_pct/high_pct percentile tails and rescale to [0, 1].
    out = np.empty_like(im)
    for c in range(im.shape[2]):
        chan = im[..., c]
        lo, hi = np.percentile(chan, [low_pct, high_pct])
        if hi <= lo:
            out[..., c] = chan
        else:
            out[..., c] = np.clip((chan - lo) / (hi - lo), 0, 1)
    return out


def auto_white_balance(im, method="grayworld"):
    # Grey-world white balance: scale each channel so its mean matches the average of all three channel means. 
    means = im.reshape(-1, 3).mean(axis=0)
    target = means.mean()
    scale = target / np.clip(means, 1e-6, None)
    out = im * scale
    return np.clip(out, 0, 1)


# 7. Full pipeline for one image
def colorize(path, out_dir, metric="ncc", features="raw", window=15,
             refine_window=2, min_size=400, do_crop=True, do_contrast=True,
             do_wb=True):
    name = os.path.splitext(os.path.basename(path))[0]
    im = skio.imread(path)
    b, g, r = split_channels(im)

    t0 = time.time()
    shift_g = align(g, b, metric=metric, features=features,
                     window=window, refine_window=refine_window,
                     min_size=min_size)
    shift_r = align(r, b, metric=metric, features=features,
                     window=window, refine_window=refine_window,
                     min_size=min_size)
    elapsed = time.time() - t0

    g_aligned = apply_shift(g, shift_g)
    r_aligned = apply_shift(r, shift_r)

    im_out = np.dstack([r_aligned, g_aligned, b])

    if do_crop:
        im_out = auto_crop(im_out)
    if do_wb:
        im_out = auto_white_balance(im_out)
    if do_contrast:
        im_out = auto_contrast(im_out)

    im_out = np.clip(im_out, 0, 1)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.jpg")
    skio.imsave(out_path, sk.img_as_ubyte(im_out))

    print(f"{name}: G shift (dy,dx)={shift_g}  R shift (dy,dx)={shift_r}  "
          f"[{elapsed:.2f}s, metric={metric}, features={features}]")
    return out_path, shift_g, shift_r

# 8. CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="image file, or a folder if --all is set")
    ap.add_argument("--all", action="store_true",
                     help="treat `input` as a folder and process every "
                          "jpg/tif/tiff/png inside it")
    ap.add_argument("--out", default="/mnt/user-data/outputs/results",
                     help="output directory")
    ap.add_argument("--metric", choices=["ncc", "l2"], default="ncc")
    ap.add_argument("--features", choices=["raw", "edge"], default="raw",
                     help="use 'edge' for plates with strong brightness "
                          "mismatch between channels, e.g. emir.tif")
    ap.add_argument("--window", type=int, default=15,
                     help="search radius in pixels: the full search window "
                          "for single-scale, or the coarsest pyramid "
                          "level's search radius for large images")
    ap.add_argument("--refine-window", type=int, default=2,
                     help="small search radius used to correct rounding "
                          "drift at each finer pyramid level (not used "
                          "in single-scale mode)")
    ap.add_argument("--min-size", type=int, default=400,
                     help="crossover size (px) below which we stop "
                          "recursing the pyramid and do a direct search")
    args = ap.parse_args()

    exts = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
    if args.all:
        files = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
                  if f.lower().endswith(exts)]
    else:
        files = [args.input]

    for f in files:
        colorize(f, args.out, metric=args.metric, features=args.features,
                  window=args.window, refine_window=args.refine_window,
                  min_size=args.min_size)


if __name__ == "__main__":
    main()
