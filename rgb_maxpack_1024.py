# =============================================================================
#  BTP — FHE Convolution-Friendly Image Compression Pipeline
#  RGB, MAXIMUM-SAVINGS EDITION — full Y+Cr+Cb packing
#  256x256 / 512x512 / 1024x1024
# =============================================================================
#
#  SETUP (run once in your PyCharm terminal):
#      pip install tenseal numpy opencv-python scipy matplotlib scikit-image
#
#  IMAGE FOLDER:
#      Create a folder called  images/  next to this script and place your
#      COLOR PNG test images inside it named as:  si_256_rgb.png
#                                                  si_512_rgb.png
#                                                  si_1024_rgb.png
#
#  BEFORE YOUR FIRST 1024px + m=8 RUN — please sanity-check this assumption
#      This file's 1024px support relies on TenSEAL automatically splitting
#      a CKKSVector into multiple physical ciphertexts when its length
#      exceeds poly_modulus_degree/2, rather than raising an error. This is
#      the same assumption already documented in the original grayscale
#      script. It could not be executed/verified in the environment this
#      file was written in (no `tenseal` package available there), so
#      before a long run, confirm it holds in YOUR environment with:
#          ctx = generate_ckks_context(20000)[0]
#          v = ts.ckks_vector(ctx, [0.0] * 20000)   # > 16384 slots at N=32768
#          print(len(v.serialize()))                 # should just work
#      If this raises instead, 1024px + m=8 combinations (see capacity
#      table below) will need a different approach than the one here.
#
#  DESIGN — WHY THIS ACHIEVES MAXIMUM SAVINGS (c ciphertexts total)
#      The previous version packed Cr+Cb together but kept Y separate
#      (2c ciphertexts total) specifically because brightening (x*1.3)
#      is WRONG to apply to chroma — it would shift color balance.
#
#      This version switches the pixel-wise operation to INVERSION
#      (x -> 255-x) instead, which removes that obstacle entirely:
#
#          True RGB negative:  R'=255-R,  G'=255-G,  B'=255-B
#          In YCbCr (BT.601, coefficients sum to 1, chroma centered at 128):
#              Y'  = 255 - Y     (exact)
#              Cb' = 256 - Cb    (exact — off by 1 from Y's constant only
#                                 because chroma's neutral point is 128,
#                                 not 0; imperceptible in practice)
#              Cr' = 256 - Cr    (exact, same reasoning)
#
#      So "x -> 255-x" is, to within a single quantization level, the
#      IDENTICAL affine transform for Y, Cb, and Cr. That means we can
#      pack ALL THREE channels into one ciphertext's SIMD slots and apply
#      ONE CMult(-1) + ONE CAdd(255) to the whole thing — no plaintext
#      mask multiply needed at all, because there's no channel-dependent
#      behaviour left to mask. The 3x3 sharpen kernel already applied
#      uniformly across channels for the same reason (a spatial filter
#      should affect every channel the same way), so it also needs zero
#      changes for full packing.
#
#      Total ciphertexts: c (ONE set, holding Y+Cr+Cb together) — exactly
#      the same ciphertext count as the original grayscale pipeline, and a
#      full 3x reduction from naive per-channel encryption.
#
#  WHY THE CONTEXT SIZE IS CHOSEN DYNAMICALLY, AND WHAT HAPPENS AT 1024px
#      Packing three channels means total slots needed = Y_blocks +
#      2 x chroma_blocks. generate_ckks_context() tries poly_modulus_degree
#      =16384 (8,192 slots) first, then 32768 (16,384 slots).
#      Full slot table (verified against the actual block-count
#      function, not hand math):
#
#        size   m   op            Y blocks  chroma(ea)  packed total  fits in
#        256    8   Pixel            1,024        256         1,536   16384(1 chunk)
#        256    8   Convolution      1,849        484         2,817   16384(1 chunk)
#        256    16  Pixel              256         64           384   16384(1 chunk)
#        256    16  Convolution        361        100           561   16384(1 chunk)
#        512    8   Pixel            4,096      1,024         6,144   16384(1 chunk)
#        512    8   Convolution      7,396      1,849        11,094   32768(1 chunk)
#        512    16  Pixel            1,024        256         1,536   16384(1 chunk)
#        512    16  Convolution      1,369        361         2,091   16384(1 chunk)
#       1024    8   Pixel           16,384      4,096        24,576   32768(2 CHUNKS)
#       1024    8   Convolution     29,241      7,396        44,033   32768(3 CHUNKS)
#       1024    16  Pixel            4,096      1,024         6,144   16384(1 chunk)
#       1024    16  Convolution      5,476      1,369         8,214   32768(1 chunk)
#
#      poly_modulus_degree is capped at 32768 (16,384 slots/chunk) — NOT
#      escalated further to 65536/131072. Mathematically, one ciphertext
#      at double the ring dimension costs EXACTLY the same bytes as two
#      TenSEAL-auto-chunked ciphertexts at the smaller dimension (size
#      scales linearly with N x #primes either way), so there's nothing
#      to gain from bigger N and real practical risk (exotic ring
#      dimensions may fall outside SEAL/TenSEAL's default parameter
#      tables). Beyond 16,384 slots, this code just lets TenSEAL's
#      automatic multi-ciphertext chunking absorb the rest.
#
#      HONEST CONSEQUENCE for m=8 at 1024px: the packed ciphertext itself
#      needs 2-3 physical chunks under the hood, so it doesn't always fit
#      in the single chunk that gave 256/512px their full 3x saving. You
#      still save real bandwidth vs. naive per-channel encryption (~25-33%
#      — see the chunk-accurate math in print_packed_bandwidth_summary),
#      just not the full 3x, because the packed total occasionally crosses
#      a chunk boundary that separate Y/Cr/Cb encryption wouldn't have hit.
#      m=16 at 1024px has no such issue — it still fits in one chunk and
#      keeps the full 3x saving, same as every 256/512px combination.
# =============================================================================

import tenseal as ts
import numpy as np
import cv2
import warnings
import logging
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from scipy.fftpack import dct, idct
import time
import matplotlib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# =============================================================================
#  USER CONFIGURATION
# =============================================================================

IMAGES_DIR = Path(__file__).parent / "images"

IMAGE_EXTENSIONS = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG",
                    ".bmp", ".BMP", ".tiff", ".TIFF"]

# =============================================================================
#  MATPLOTLIB BACKEND
# =============================================================================
matplotlib.use('TkAgg')   # <-- change to 'MacOSX' on macOS if needed

# =============================================================================
#  SUPPRESS TENSEAL / NUMPY WARNINGS
# =============================================================================

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.getLogger('tenseal').setLevel(logging.ERROR)

_dct_cache   = {}
_quant_cache = {}


@contextmanager
def silence_stdout_stderr():
    """Redirect stdout and stderr to /dev/null (suppresses TenSEAL C++ output)."""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull), redirect_stdout(fnull):
            yield


# =============================================================================
#  IMAGE LOADING — COLOR
# =============================================================================

def load_color_image(image_size: int) -> np.ndarray:
    """Load a COLOR image from images/si_{image_size}_rgb.<ext>. Returns BGR uint8."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    stem = f"si_{image_size}_rgb"
    for ext in IMAGE_EXTENSIONS:
        candidate = IMAGES_DIR / f"{stem}{ext}"
        if candidate.exists():
            img = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"cv2 could not decode: {candidate}")
            print(f"[Image] Loaded '{candidate.name}'  shape={img.shape}  dtype={img.dtype}")
            return img

    raise FileNotFoundError(
        f"\n[Error] No image found for size {image_size}.\n"
        f"  Expected file : {IMAGES_DIR / stem}.<png|jpg|...>\n"
        f"  Images folder : {IMAGES_DIR.resolve()}\n"
        f"  Files present : {[f.name for f in IMAGES_DIR.iterdir()] if IMAGES_DIR.exists() else '(folder missing)'}\n"
        f"\n  Fix: place a COLOR PNG named  si_{image_size}_rgb.png  in the images/ folder."
    )


def split_and_subsample_chroma(bgr_image: np.ndarray):
    """BGR -> YCrCb, subsample Cr/Cb to half resolution (4:2:0). Y stays full-res."""
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    h, w = y.shape
    cr_small = cv2.resize(cr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    cb_small = cv2.resize(cb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return y, cr_small, cb_small


def merge_and_upsample_chroma(y_plane: np.ndarray, cr_plane: np.ndarray,
                               cb_plane: np.ndarray) -> np.ndarray:
    """Upsample Cr/Cb back to Y's resolution and convert back to BGR."""
    h, w = y_plane.shape
    cr_full = cv2.resize(cr_plane, (w, h), interpolation=cv2.INTER_LINEAR)
    cb_full = cv2.resize(cb_plane, (w, h), interpolation=cv2.INTER_LINEAR)
    ycrcb = cv2.merge([y_plane, cr_full, cb_full])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# =============================================================================
#  BLOCK-COUNT PREDICTION (needed BEFORE encryption, to size the context)
# =============================================================================

def compute_block_count(h: int, w: int, m: int, use_overlap_trick: bool) -> int:
    """
    Predict how many m x m blocks a plane of size (h, w) will produce,
    WITHOUT doing any of the actual DCT work. Mirrors the exact loop
    bounds used in compress_plane() / reconstruct_plane_from_lanes(), so
    the slot count this returns is always exactly right — computed via
    Python's own range length rather than a hand-derived formula, to
    avoid any off-by-one mismatch with the real compression loop.
    """
    step = (m - 2) if use_overlap_trick else m
    pad_h = (step - h % step) % step
    pad_w = (step - w % step) % step
    if use_overlap_trick:
        pad_h += 2
        pad_w += 2
    padded_h, padded_w = h + pad_h, w + pad_w
    n_i = len(range(0, padded_h - (m - step), step))
    n_j = len(range(0, padded_w - (m - step), step))
    return n_i * n_j


# =============================================================================
#  CKKS CONTEXT — DYNAMICALLY SIZED, 256/512 ONLY
# =============================================================================

def generate_ckks_context(total_slots_needed: int):
    """
    Build a TenSEAL CKKS context sized to fit total_slots_needed SIMD slots.

    poly_modulus_degree=16384 -> 8,192  slots/chunk (tried first)
    poly_modulus_degree=32768 -> 16,384 slots/chunk (used above that)

    Capped at 32768 — NOT escalated further. See the file-header capacity
    table: beyond 16,384 slots (only 1024px + m=8, in this file's
    supported range), TenSEAL's automatic multi-ciphertext chunking
    absorbs the remainder. This costs bytes proportionally (more physical
    ciphertexts under the hood) but needs no larger ring dimension —
    mathematically equivalent bytes either way, and far more portable.

    Using a larger N with the SAME modulus chain is only ever AS SECURE
    or MORE secure (never less) at a fixed total modulus bit-length, so
    bumping to 32768 is safe from a security standpoint regardless.

    Returns
    -------
    (context, poly_modulus_degree_used)

    The 9-prime chain "coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60]",
    provides 7 multiplicative depth levels.
    """
    coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 40, 40, 40, 60]

    if total_slots_needed <= 8192:
        poly_degree = 16384
    else:
        poly_degree = 32768
        if total_slots_needed <= 16384:
            print(f"  [Setup] {total_slots_needed} slots needed > 8192 — "
                  f"bumping poly_modulus_degree to 32768 for this run.")
        else:
            n_chunks = -(-total_slots_needed // 16384)  # ceil
            print(f"  [Setup] {total_slots_needed} slots needed > 16384 — "
                  f"each logical ciphertext will need ~{n_chunks} physical "
                  f"chunks under TenSEAL's automatic splitting (see the "
                  f"file-header capacity table for the bandwidth impact).")

    with silence_stdout_stderr():
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        )
        context.global_scale = 2 ** 40

    return context, poly_degree


# =============================================================================
#  HELPER FUNCTIONS (unchanged from the grayscale pipeline)
# =============================================================================

def compute_ssi(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Structural Similarity Index (Definition 2 in the paper), via skimage."""
    return ssim(original, reconstructed, data_range=255)


def compute_compression_ratio(m: int, c: int, use_overlap_trick: bool):
    """Compression ratio B0:B1 (Definition 3 in the paper), per-block."""
    B0 = (m - 2) ** 2 if use_overlap_trick else m ** 2
    B1 = c
    return B0, B1, (B1 / B0) * 100


def get_dct_matrix(m: int) -> np.ndarray:
    """Returns (and caches) the m×m orthonormal DCT-II matrix T_m."""
    if m not in _dct_cache:
        T = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i == 0:
                    T[i, j] = 1 / np.sqrt(m)
                else:
                    T[i, j] = np.sqrt(2 / m) * np.cos((2 * j + 1) * i * np.pi / (2 * m))
        _dct_cache[m] = T
    return _dct_cache[m]


def get_quantization_matrix(m: int, quality: int) -> np.ndarray:
    """Returns (and caches) the quality-scaled JPEG luminance Q matrix."""
    key = (m, quality)
    if key in _quant_cache:
        return _quant_cache[key]

    if quality > 50:
        scale = (100 - quality) / 50.0
    elif quality < 50:
        scale = 50.0 / quality
    else:
        scale = 1.0

    if m == 8:
        base = np.array([
            [16, 11, 10, 16,  24,  40,  51,  61],
            [12, 12, 14, 19,  26,  58,  60,  55],
            [14, 13, 16, 24,  40,  57,  69,  56],
            [14, 17, 22, 29,  51,  87,  80,  62],
            [18, 22, 37, 56,  68, 109, 103,  77],
            [24, 35, 55, 64,  81, 104, 113,  92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103,  99],
        ], dtype=np.float64)
    elif m == 16:
        q8 = get_quantization_matrix(8, 50)
        base = np.array(
            [[q8[i // 2, j // 2] for j in range(16)] for i in range(16)],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unsupported block size m={m}. Use 8 or 16.")

    scaled = np.maximum(np.floor(base * scale + 0.5), 1.0)
    _quant_cache[key] = scaled
    return scaled


def build_zigzag_coordinates(m: int):
    """Returns the (row, col) visit order for an m×m zigzag scan."""
    indices = []
    for diag in range(2 * m - 1):
        if diag % 2 == 0:
            x = min(diag, m - 1)
            y = diag - x
            while x >= 0 and y < m:
                indices.append((x, y))
                x -= 1
                y += 1
        else:
            y = min(diag, m - 1)
            x = diag - y
            while y >= 0 and x < m:
                indices.append((x, y))
                x += 1
                y -= 1
    return indices


def execute_zigzag_scan(matrix: np.ndarray, m: int) -> np.ndarray:
    """Flatten a 2D block into a 1D vector in zigzag order."""
    coords = build_zigzag_coordinates(m)
    return np.array([matrix[x, y] for x, y in coords])


def execute_inverse_zigzag(vector: np.ndarray, m: int) -> np.ndarray:
    """Reconstruct a 2D block from a 1D zigzag-ordered vector."""
    coords = build_zigzag_coordinates(m)
    matrix = np.zeros((m, m), dtype=np.float64)
    for idx, (x, y) in enumerate(coords):
        matrix[x, y] = vector[idx]
    return matrix


def zigzag_indices(m: int):
    """Alternative zigzag coordinate generator (used in the server functions)."""
    indices = []
    for d in range(2 * m - 1):
        for i in range(max(0, d - m + 1), min(d + 1, m)):
            j = d - i
            indices.append((j, i) if d % 2 == 0 else (i, j))
    return indices


def measure_bandwidth(ciphertexts, description: str = "Data") -> int:
    """Print and return the total serialized size in bytes of a ciphertext list."""
    total_bytes = sum(len(ct.serialize()) for ct in ciphertexts)
    mb = total_bytes / (1024 * 1024)
    print(f"  [{description}] Bandwidth: {total_bytes:,} bytes  ({mb:.3f} MB)")
    return total_bytes


def print_packed_bandwidth_summary(total_bytes: int, direction: str,
                                    packed_chunks: int = 1, naive_chunks: int = 3):
    """
    Report bandwidth for the single packed (Y+Cr+Cb) ciphertext set, and
    the saving vs. naive 3-channel encryption.

    packed_chunks / naive_chunks let this stay ACCURATE once TenSEAL's
    automatic multi-ciphertext chunking kicks in (see generate_ckks_context
    and the file-header capacity table). At 256/512px (and 1024px+m=16)
    both are effectively 1:3 and this reduces to the clean "3x" story.
    At 1024px+m=8, packing and naive-separate encryption land on
    DIFFERENT chunk counts (e.g. 2 vs 3, or 3 vs 4) because the packed
    total occasionally crosses a chunk boundary that separate per-channel
    encryption wouldn't have — a flat "assume 3x" comparison would be
    wrong there, so this uses the real chunk ratio instead.
    """
    naive_bytes = total_bytes * (naive_chunks / packed_chunks)
    saving_pct = 100 * (1 - total_bytes / naive_bytes)
    print(f"\n  [{direction}] Packed Y+Cr+Cb (c ciphertexts, "
          f"{packed_chunks} chunk{'s' if packed_chunks != 1 else ''}/ciphertext): "
          f"{total_bytes:,} bytes  ({total_bytes / (1024*1024):.3f} MB)")
    print(f"  [{direction}] vs. naive separate Y/Cr/Cb encryption "
          f"({naive_chunks} chunks/ciphertext, ~{naive_bytes:,.0f} bytes): "
          f"{total_bytes/naive_bytes:.2f}x  ({saving_pct:.1f}% smaller)")


# =============================================================================
#  CLIENT — COMPRESS A SINGLE PLANE (shared by Y, Cr, Cb)
# =============================================================================

def compress_plane(plane: np.ndarray, m: int, c: int, quality: int,
                    use_overlap_trick: bool):
    """Block + level + DCT + quantize + zigzag + cutoff for ONE plane."""
    original_h, original_w = plane.shape
    Q    = get_quantization_matrix(m, quality)
    step = (m - 2) if use_overlap_trick else m

    pad_h = (step - original_h % step) % step
    pad_w = (step - original_w % step) % step
    if use_overlap_trick:
        pad_h += 2
        pad_w += 2

    working_image = np.pad(plane, ((0, pad_h), (0, pad_w)), mode='reflect')
    padded_h, padded_w = working_image.shape

    compressed_payload = []
    for i in range(0, padded_h - (m - step), step):
        for j in range(0, padded_w - (m - step), step):
            block = working_image[i:i+m, j:j+m].astype(np.float64)
            leveled       = block - 128.0
            dct_freqs     = dct(dct(leveled, axis=0, norm='ortho'), axis=1, norm='ortho')
            quantized     = np.round(dct_freqs / Q)
            linear_vector = execute_zigzag_scan(quantized, m)
            compressed_payload.append(linear_vector[:c])

    compressed_payload = np.array(compressed_payload)   # (N_blocks, c)
    return compressed_payload, (original_h, original_w)


def encrypt_packed_triple(y_payload: np.ndarray, cr_payload: np.ndarray,
                           cb_payload: np.ndarray, context, c: int):
    """
    Pack ALL THREE planes into the SAME ciphertext's SIMD slots, per
    frequency lane: [Y_lane || Cr_lane || Cb_lane]. Still only `c`
    ciphertexts total, for all three channels combined — this is the
    maximum-savings scheme.
    """
    encrypted = []
    for fi in range(c):
        lane = (y_payload[:, fi].tolist()
                + cr_payload[:, fi].tolist()
                + cb_payload[:, fi].tolist())
        with silence_stdout_stderr():
            encrypted.append(ts.ckks_vector(context, lane))
    return encrypted


def client_compress_and_encrypt_rgb(bgr_image: np.ndarray, context, m: int,
                                     c: int, quality: int,
                                     use_overlap_trick: bool):
    """
    Full client-side RGB compression + encryption, all three channels
    packed into ONE ciphertext set.

    Returns
    -------
    packed_cts : list[CKKSVector]  length c  (Y+Cr+Cb packed together)
    meta       : dict with everything needed to reconstruct the image
    """
    print("--- Client: Splitting color image into Y / Cr / Cb ---")
    y_plane, cr_plane, cb_plane = split_and_subsample_chroma(bgr_image)
    print(f"  [Client] Y  plane: {y_plane.shape}   "
          f"Cr/Cb planes (subsampled 4:2:0): {cr_plane.shape}")

    print("\n--- Client: Compressing Y, Cr, Cb ---")
    y_payload,  y_shape      = compress_plane(y_plane,  m, c, quality, use_overlap_trick)
    cr_payload, chroma_shape = compress_plane(cr_plane, m, c, quality, use_overlap_trick)
    cb_payload, _            = compress_plane(cb_plane, m, c, quality, use_overlap_trick)
    print(f"  [Client] Y : {y_payload.shape[0]} blocks × {c} coefficients")
    print(f"  [Client] Cr/Cb: {cr_payload.shape[0]} blocks each × {c} coefficients")
    print(f"  [Client] Packed slots per ciphertext: "
          f"{y_payload.shape[0] + 2 * cr_payload.shape[0]}")

    print("\n--- Client: Encrypting Y+Cr+Cb packed (c ciphertexts TOTAL) ---")
    packed_cts = encrypt_packed_triple(y_payload, cr_payload, cb_payload, context, c)

    meta = {
        'orig_hw':       bgr_image.shape[:2],
        'y_shape':       y_shape,
        'chroma_shape':  chroma_shape,
        'y_blocks':      y_payload.shape[0],
        'chroma_blocks': cr_payload.shape[0],   # Cr and Cb always match
    }
    return packed_cts, meta


# =============================================================================
#  SERVER — HOMOMORPHIC DECOMPRESS  (unchanged from grayscale pipeline)
# =============================================================================

def server_homomorphic_decompress(encrypted_cts, context, m: int, c: int,
                                   num_blocks: int, quality: int):
    """Two-pass separable IDCT over CKKS ciphertexts."""
    print("\n--- Server: Homomorphically Decompressing ---")
    Q       = get_quantization_matrix(m, quality)
    T       = get_dct_matrix(m)
    T_T     = T.T
    zz_idx  = zigzag_indices(m)
    T_list  = T.tolist()
    TT_list = T_T.tolist()

    grid = [[None] * m for _ in range(m)]
    for idx in range(min(len(encrypted_cts), c)):
        r, col = zz_idx[idx]
        grid[r][col] = encrypted_cts[idx] * float(Q[r, col])

    intermediate = [[None] * m for _ in range(m)]
    for k in range(m):
        for l in range(m):
            ct_kl = grid[k][l]
            if ct_kl is None:
                continue
            T_row = T_list[l]
            for j in range(m):
                contrib = ct_kl * T_row[j]
                intermediate[k][j] = (contrib if intermediate[k][j] is None
                                      else intermediate[k][j] + contrib)

    final_grid = [[None] * m for _ in range(m)]
    for k in range(m):
        for j in range(m):
            ct_kj = intermediate[k][j]
            if ct_kj is None:
                continue
            for i in range(m):
                contrib = ct_kj * TT_list[i][k]
                final_grid[i][j] = (contrib if final_grid[i][j] is None
                                    else final_grid[i][j] + contrib)

    for i in range(m):
        for j in range(m):
            if final_grid[i][j] is None:
                final_grid[i][j] = ts.ckks_vector(context, [128.0] * num_blocks)
            else:
                final_grid[i][j] = final_grid[i][j] + 128.0

    return final_grid


# =============================================================================
#  SERVER — PROCESSING OPERATIONS
# =============================================================================

def server_process_inversion(decompressed_grid, m: int):
    """
    Apply pixel-wise inversion (x -> 255 - x) homomorphically.

    Chosen specifically because color inversion is, to within a single
    quantization level, the SAME affine transform for Y, Cb, AND Cr (see
    the file-header derivation) — so it applies correctly to a ciphertext
    that packs all three channels together, with no per-channel masking.
    """
    print("\n--- Server: Applying pixel-wise inversion ---")
    for i in range(m):
        for j in range(m):
            if isinstance(decompressed_grid[i][j], ts.CKKSVector):
                decompressed_grid[i][j] = decompressed_grid[i][j] * -1.0 + 255.0
    return decompressed_grid


def server_process_convolution(decompressed_grid, m: int):
    """
    Apply a 3×3 sharpening kernel homomorphically. Already channel-
    agnostic — a spatial filter should affect every packed channel the
    same way, so this needs no changes for full Y+Cr+Cb packing either.
    """
    print("\n--- Server: Applying 3×3 convolutional sharpening filter ---")
    kernel = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    processed_grid = [[None] * m for _ in range(m)]

    for i in range(1, m - 1):
        for j in range(1, m - 1):
            val = decompressed_grid[i][j] * 0.0
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    w = kernel[ki + 1][kj + 1]
                    if w != 0:
                        val += decompressed_grid[i + ki][j + kj] * float(w)
            processed_grid[i][j] = val

    for i in range(m):
        for j in range(m):
            if processed_grid[i][j] is None:
                processed_grid[i][j] = decompressed_grid[i][j] * 1.0

    return processed_grid


def server_process(decompressed_grid, m: int, operation: str):
    """
    Apply the SAME operation across the whole packed (Y+Cr+Cb) grid.
    No masking needed for either supported operation — see the two
    functions above for why each is channel-agnostic.
    """
    if operation == 'Pixel':
        return server_process_inversion(decompressed_grid, m)
    else:
        return server_process_convolution(decompressed_grid, m)


# =============================================================================
#  SERVER — HOMOMORPHIC RECOMPRESS  (unchanged from grayscale pipeline)
# =============================================================================

def server_homomorphic_compress(processed_grid, m: int, c: int, context, quality: int):
    """Two-pass separable DCT over CKKS ciphertexts."""
    print("\n--- Server: Homomorphically Re-compressing ---")
    num_blocks = processed_grid[0][0].size()
    Q       = get_quantization_matrix(m, quality)
    T       = get_dct_matrix(m)
    T_T     = T.T
    zz_idx  = zigzag_indices(m)
    T_list  = T.tolist()
    TT_list = T_T.tolist()

    for i in range(m):
        for j in range(m):
            if isinstance(processed_grid[i][j], ts.CKKSVector):
                processed_grid[i][j] -= 128.0

    intermediate = [[None] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            ct_ij = processed_grid[i][j]
            if not isinstance(ct_ij, ts.CKKSVector):
                continue
            for v in range(m):
                contrib = ct_ij * TT_list[j][v]
                intermediate[i][v] = (contrib if intermediate[i][v] is None
                                      else intermediate[i][v] + contrib)

    recompressed_cts = []
    for k in range(c):
        u, v      = zz_idx[k]
        T_row_u   = T_list[u]
        coeff_sum = None
        for i in range(m):
            ct_iv = intermediate[i][v]
            if ct_iv is None:
                continue
            contrib   = ct_iv * T_row_u[i]
            coeff_sum = (contrib if coeff_sum is None else coeff_sum + contrib)

        if coeff_sum is not None:
            recompressed_cts.append(coeff_sum * float(1.0 / Q[u, v]))
        else:
            recompressed_cts.append(ts.ckks_vector(context, [0.0] * num_blocks))

    return recompressed_cts


# =============================================================================
#  CLIENT — DECRYPT AND DECOMPRESS
# =============================================================================

def reconstruct_plane_from_lanes(decrypted_lanes: np.ndarray, orig_shape,
                                  m: int, c: int, quality: int,
                                  use_overlap_trick: bool) -> np.ndarray:
    """Reconstruct one plane from an already-decrypted (N_blocks, c) array."""
    original_h, original_w = orig_shape
    step = (m - 2) if use_overlap_trick else m

    pad_h = (step - original_h % step) % step
    pad_w = (step - original_w % step) % step
    if use_overlap_trick:
        pad_h += 2
        pad_w += 2

    padded_h = original_h + pad_h
    padded_w = original_w + pad_w
    Q        = get_quantization_matrix(m, quality)

    reconstructed = np.zeros((padded_h, padded_w), dtype=np.float64)

    block_idx = 0
    for i in range(0, padded_h - (m - step), step):
        for j in range(0, padded_w - (m - step), step):
            vec       = np.zeros(m * m, dtype=np.float64)
            vec[:c]   = decrypted_lanes[block_idx]
            q_block   = execute_inverse_zigzag(vec, m)
            dct_freqs = q_block * Q
            spatial   = idct(idct(dct_freqs, axis=0, norm='ortho'),
                             axis=1, norm='ortho') + 128.0
            if use_overlap_trick:
                reconstructed[i+1:i+m-1, j+1:j+m-1] = spatial[1:m-1, 1:m-1]
            else:
                reconstructed[i:i+m, j:j+m] = spatial
            block_idx += 1

    reconstructed = np.clip(reconstructed, 0, 255)
    if use_overlap_trick:
        return reconstructed[1:original_h+1, 1:original_w+1].astype(np.uint8)
    else:
        return reconstructed[:original_h, :original_w].astype(np.uint8)


def client_decrypt_packed_triple(encrypted_vectors, y_shape, chroma_shape,
                                  y_blocks: int, chroma_blocks: int, m: int,
                                  c: int, quality: int, use_overlap_trick: bool):
    """
    Decrypt c ciphertexts, each holding [Y_lane || Cr_lane || Cb_lane] in
    its slots, split by the known block-count boundaries, and reconstruct
    all three planes.
    """
    decrypted = np.array([ct.decrypt() for ct in encrypted_vectors])  # (c, y_blocks+2*chroma_blocks)

    y_lanes  = decrypted[:, :y_blocks].T                                    # (y_blocks, c)
    cr_lanes = decrypted[:, y_blocks:y_blocks + chroma_blocks].T            # (chroma_blocks, c)
    cb_lanes = decrypted[:, y_blocks + chroma_blocks:].T                    # (chroma_blocks, c)

    y_plane  = reconstruct_plane_from_lanes(y_lanes,  y_shape,      m, c, quality, use_overlap_trick)
    cr_plane = reconstruct_plane_from_lanes(cr_lanes, chroma_shape, m, c, quality, use_overlap_trick)
    cb_plane = reconstruct_plane_from_lanes(cb_lanes, chroma_shape, m, c, quality, use_overlap_trick)
    return y_plane, cr_plane, cb_plane


# =============================================================================
#  SUMMARY PRINTING
# =============================================================================

def print_pipeline_summary(image_size, operation, m, c, use_overlap_trick,
                            y_original, y_reconstructed, total_time,
                            bytes_out, bytes_in, packed_chunks, naive_chunks):
    ssi_val = compute_ssi(y_original, y_reconstructed)
    _, _, ratio_pct = compute_compression_ratio(m, c, use_overlap_trick)

    print("\n" + "=" * 60)
    print(f"  {image_size}x{image_size} RGB (full pack)  |  {operation}  |  m={m}  |  c={c}")
    print("=" * 60)
    print(f"  SSI (Y channel)      : {ssi_val:.3f}")
    print(f"  Compression ratio    : 100:{ratio_pct:.1f}")
    print(f"  Total pipeline time  : {total_time:.2f} s")
    print_packed_bandwidth_summary(bytes_out, "Client -> Server", packed_chunks, naive_chunks)
    print_packed_bandwidth_summary(bytes_in,  "Server -> Client", packed_chunks, naive_chunks)


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def main_execution_pipeline():
    """Interactive end-to-end FHE RGB image compression pipeline (full packing)."""

    print("=" * 60)
    print("  FHE Convolution-Friendly Image Compression Pipeline")
    print("  RGB — Maximum-Savings Edition (Y+Cr+Cb packed, c ciphertexts)")
    print("=" * 60)
    print()

    # ── User inputs ───────────────────────────────────────────────────────
    image_size = int(input("Image size (256 / 512 / 1024): "))
    if image_size not in (256, 512, 1024):
        raise ValueError("This pipeline only supports 256, 512, or 1024.")

    operation = input("Operation (Convolution / Pixel): ").strip()
    if operation not in ('Convolution', 'Pixel'):
        raise ValueError("Operation must be 'Convolution' or 'Pixel'.")
    use_overlap = (operation == 'Convolution')

    macro_block = int(input("Block size m (8 / 16): "))
    if macro_block not in (8, 16):
        raise ValueError("Block size must be 8 or 16.")

    cutoff_map = {
        ('Convolution', 8): 30, ('Pixel', 8): 22,
        ('Convolution', 16): 70, ('Pixel', 16): 63,
    }
    cutoff = cutoff_map[(operation, macro_block)]

    quality = int(input(
        "Scaling quality q [1-99]  "
        "(50=identity | >50=preserve quality | <50=more compression): "
    ))

    print()

    # ── Load image ────────────────────────────────────────────────────────
    bgr_image = load_color_image(image_size)   # (H, W, 3) uint8, BGR

    # ── Predict slot requirement BEFORE building the context ───────────────
    y_blocks_needed      = compute_block_count(image_size, image_size, macro_block, use_overlap)
    chroma_blocks_needed = compute_block_count(image_size // 2, image_size // 2, macro_block, use_overlap)
    total_slots_needed   = y_blocks_needed + 2 * chroma_blocks_needed
    print(f"[Setup] Predicted slots needed (Y + 2xchroma): {total_slots_needed}")

    # ── CKKS context (sized to fit, per the file-header explanation) ───────
    print("[Setup] Generating CKKS context …")
    context, poly_degree = generate_ckks_context(total_slots_needed)
    print(f"[Setup] Context ready (poly_modulus_degree={poly_degree}).\n")

    # ── Chunk-accurate bandwidth bookkeeping (see print_packed_bandwidth_summary) ─
    slots_per_chunk = poly_degree // 2
    y_chunks        = -(-y_blocks_needed // slots_per_chunk)       # ceil
    chroma_chunks   = -(-chroma_blocks_needed // slots_per_chunk)  # ceil
    naive_chunks    = y_chunks + 2 * chroma_chunks
    packed_chunks   = -(-total_slots_needed // slots_per_chunk)    # ceil
    if packed_chunks > 1 or naive_chunks != 3:
        print(f"[Setup] Chunk accounting: packed={packed_chunks} chunk(s)/ciphertext, "
              f"naive-separate=Y:{y_chunks}+Cr:{chroma_chunks}+Cb:{chroma_chunks}"
              f"={naive_chunks} chunk(s)/ciphertext\n")

    pipeline_t0 = time.time()

    # ── Client: compress + encrypt (Y+Cr+Cb packed into ONE ciphertext set) ─
    t0 = time.time()
    packed_cts, meta = client_compress_and_encrypt_rgb(
        bgr_image, context, m=macro_block, c=cutoff, quality=quality,
        use_overlap_trick=use_overlap,
    )
    print(f"\n  [Time] Compress + Encrypt : {time.time() - t0:.2f} s")

    print("\n[Bandwidth] Client -> Server:")
    bytes_out = measure_bandwidth(packed_cts, description="Packed Y+Cr+Cb")
    print_packed_bandwidth_summary(bytes_out, "Client -> Server", packed_chunks, naive_chunks)

    # ── Server: decompress -> process -> recompress (ONE call each) ────────
    t0 = time.time()
    decompressed = server_homomorphic_decompress(
        packed_cts, context, macro_block, cutoff, meta['y_blocks'] + 2 * meta['chroma_blocks'], quality)
    processed = server_process(decompressed, macro_block, operation)
    recompressed = server_homomorphic_compress(
        processed, macro_block, cutoff, context, quality)
    print(f"\n  [Time] Server Homomorphic Processing : {time.time() - t0:.2f} s")

    print("\n[Bandwidth] Server -> Client:")
    bytes_in = measure_bandwidth(recompressed, description="Packed Y+Cr+Cb")
    print_packed_bandwidth_summary(bytes_in, "Server -> Client", packed_chunks, naive_chunks)

    # ── Client: decrypt + decompress + merge ────────────────────────────────
    t0 = time.time()
    y_result, cr_result, cb_result = client_decrypt_packed_triple(
        recompressed, meta['y_shape'], meta['chroma_shape'],
        meta['y_blocks'], meta['chroma_blocks'],
        macro_block, cutoff, quality, use_overlap,
    )
    result_bgr = merge_and_upsample_chroma(y_result, cr_result, cb_result)
    print(f"\n  [Time] Decrypt + Decompress + Merge : {time.time() - t0:.2f} s")

    total_time = time.time() - pipeline_t0

    # ── Reference Y plane for SSI (from the original, uncompressed image) ──
    y_original, _, _ = split_and_subsample_chroma(bgr_image)

    print_pipeline_summary(
        image_size, operation, macro_block, cutoff, use_overlap,
        y_original, y_result, total_time, bytes_out, bytes_in,
        packed_chunks, naive_chunks,
    )

    # ── Display ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"FHE RGB Pipeline (full pack) — {image_size}×{image_size}  |  "
        f"{operation}  |  m={macro_block}  |  c={cutoff}",
        fontsize=12,
    )
    axes[0].imshow(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title("FHE Processed")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("  Pipeline completed successfully.")
    print("=" * 60)


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    main_execution_pipeline()


# =============================================================================
#  NOTES
# =============================================================================
#
#  [1] Use PNG source images, not JPEG — same reasoning as before: JPEG's
#      own DCT + 4:2:0 chroma subsampling would compound with this
#      pipeline's own compression and subsampling steps.
#
#  [2] Precision note on inversion
#      Y'=255-Y is exact. Cb'=256-Cb and Cr'=256-Cr are also exact under
#      BT.601 — but this code uses the SAME constant (255) for all three,
#      trading a single quantization level of chroma accuracy (invisible
#      here, given the pipeline already quantizes far more aggressively
#      via the DCT cutoff) for a genuinely identical, unmasked operation
#      across all packed channels. If exact chroma accuracy mattered more
#      than simplicity, the CAdd step could use a length-matching plain
#      vector ([255]*y_blocks + [256]*chroma_blocks + [256]*chroma_blocks)
#      instead of a scalar — CAdd doesn't consume a multiplicative level,
#      so this would be free to add later without touching the CMult(-1).
#
#  [3] Why brightening was dropped
#      Brightening (x*1.3) is NOT the same operation for luma and chroma —
#      scaling chroma shifts color balance. Inversion has no such
#      asymmetry (see file header), which is why it was chosen as the
#      pixel-wise operation for this full-packing version.
#
#  [4] Why the context size is chosen per-run, not fixed
#      See the file-header capacity table. generate_ckks_context() checks
#      the actual slot requirement first and only pays the bigger-
#      ciphertext cost (N=32768) when necessary, capping there and
#      relying on TenSEAL's automatic chunking beyond that — the same
#      auto-scaling idea found in the reference C++ code, just handled
#      without ever needing an exotic ring dimension.
#
#  [5] Bandwidth, end to end
#      At 256px, 512px, and 1024px+m=16: c ciphertexts total for Y+Cr+Cb
#      combined, fitting in a single physical chunk — the same count AND
#      the same physical size as the ORIGINAL GRAYSCALE pipeline, a full
#      3x reduction from naive per-channel encryption. Going grayscale ->
#      naive RGB (3x) -> Y-separate/chroma-packed (2x) -> this version
#      (1x, i.e. matching grayscale) is a clean progression worth showing
#      side by side in a report.
#
#  [6] 1024px + m=8 is the one case that doesn't reach the full 3x
#      At this size/block-size combination, even the fully-packed
#      ciphertext needs 2 (Pixel) or 3 (Convolution) physical chunks
#      under TenSEAL's automatic splitting — see the file-header capacity
#      table. You still save real bandwidth over naive per-channel
#      encryption (33% for Pixel, 25% for Convolution — computed exactly,
#      not assumed, via the packed_chunks/naive_chunks accounting in
#      main_execution_pipeline and print_packed_bandwidth_summary), but
#      not the full 3x, because the packed total occasionally crosses a
#      chunk boundary that separate Y/Cr/Cb encryption wouldn't have hit.
#      If you specifically need the full 3x at 1024px, use m=16 instead
#      of m=8 — it stays within a single chunk and keeps the full saving.
# =============================================================================
