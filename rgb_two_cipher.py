# =============================================================================
#  BTP — FHE Convolution-Friendly Image Compression Pipeline (RGB EDITION)
#  256x256 and 512x512 color images only
# =============================================================================
#
#  SETUP (run once in your PyCharm terminal):
#      pip install tenseal numpy opencv-python scipy matplotlib scikit-image
#
#  IMAGE FOLDER:
#      Create a folder called  images/  next to this script and place your
#      COLOR PNG test images inside it named as:  si_256_rgb.png
#                                                  si_512_rgb.png
#      PNG is recommended over JPEG — see NOTES at the bottom of this file.
#
#  HOW TO RUN IN PYCHARM:
#      Right-click → "Run btp_local_rgb.py"  or  Shift+F10
#      The terminal at the bottom will prompt for your choices interactively.
#
#  WHY ONLY 256 / 512?
#      generate_ckks_context() below is deliberately restricted to these two
#      sizes. At m=8 with the convolution overlap trick, 512x512 already uses
#      ~7,400 of the 8,192 SIMD slots available at poly_modulus_degree=16384
#      (see the capacity notes in generate_ckks_context). 1024/2048 would
#      overflow this context and need a deeper/larger one — out of scope here.
#
#  RGB DESIGN — READ THIS BEFORE MODIFYING
#      Naively encrypting R, G, B as three independent grayscale pipelines
#      would triple bandwidth (3 x c ciphertexts instead of c). Instead:
#        1. Convert to YCrCb and subsample Cr/Cb to half resolution (4:2:0),
#           exactly like the JPEG standard this whole design is based on.
#        2. Encrypt Y as its own set of c ciphertexts (unchanged from the
#           grayscale pipeline).
#        3. PACK Cr and Cb into a SECOND set of c ciphertexts — each
#           ciphertext's SIMD slots hold [Cr_blocks || Cb_blocks]
#           concatenated. This costs nothing extra per ciphertext (CKKS
#           ciphertext byte size depends on poly_modulus_degree and the
#           modulus chain, NOT on how many slots are populated) and exploits
#           capacity your context already pays for but the grayscale
#           pipeline left unused.
#      Total bandwidth: c (Y) + c (Cr+Cb packed) = 2c ciphertexts,
#      vs. 3c for the naive approach — a real ~33% reduction, not a hack.
#
#      Brightening (pixel-wise) is applied to Y ONLY — brightening should
#      adjust luma, not shift color balance by also scaling chroma.
#      Sharpening (convolution) is applied to BOTH Y and the packed Cr/Cb
#      set, since a spatial filter should normally affect all channels.
#      Both server_process_pixelwise / server_process_convolution functions
#      are UNCHANGED from the grayscale version — they operate blindly on
#      whatever ciphertext they're handed, so packing two channels into one
#      ciphertext's slots is transparent to them (CMult/CAdd are per-slot
#      operations; slots never mix with each other).
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
#  USER CONFIGURATION — edit these two lines only
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
    """
    Load a COLOR image from the local images/ folder.

    Looks for:  images/si_{image_size}_rgb.<ext>

    Returns
    -------
    np.ndarray   shape (H, W, 3), dtype uint8, BGR order (OpenCV convention)
    """
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
    """
    BGR -> YCrCb, then subsample Cr/Cb to half resolution (4:2:0), matching
    the JPEG standard this pipeline is inspired by. Y stays full resolution.

    Returns
    -------
    y_plane  : (H, W)       uint8
    cr_plane : (H//2, W//2) uint8
    cb_plane : (H//2, W//2) uint8
    """
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
#  CKKS CONTEXT — RESTRICTED TO 256 / 512
# =============================================================================

def generate_ckks_context(image_size: int) -> ts.Context:
    """
    Build a TenSEAL CKKS context. ONLY 256x256 and 512x512 are supported.

    poly_modulus_degree=16384 -> 8,192 usable SIMD slots.

    Capacity check (worst case: m=8, convolution/overlap trick):
        256px -> ~1,849  Y blocks   (well within 8,192)
        512px -> ~7,396  Y blocks   (fits, ~90% utilised)
        chroma (half-res, packed Cr+Cb) is always well under capacity
        because it operates on a quarter of the pixels of Y.

    1024/2048 are intentionally NOT supported here — at m=8 they would
    need on the order of 16k-30k+ Y blocks, which overflows this context
    and would require a deeper parameter set (more primes -> more
    bandwidth), reopening the trade-off discussed for the grayscale case.

    The 9-prime chain "coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60]",
    provides 7 multiplicative depth levels.
    """
    if image_size not in (256, 512):
        raise ValueError(
            f"Unsupported image_size={image_size}. This RGB pipeline only "
            f"supports 256 or 512 (see generate_ckks_context docstring)."
        )

    with silence_stdout_stderr():
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60],
        )
        context.global_scale = 2 ** 40

    return context


# =============================================================================
#  HELPER FUNCTIONS (unchanged from the grayscale pipeline)
# =============================================================================

def compute_ssi(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Structural Similarity Index (Definition 2 in the paper), via skimage."""
    return ssim(original, reconstructed, data_range=255)


def compute_compression_ratio(m: int, c: int, use_overlap_trick: bool):
    """
    Compression ratio B0:B1 (Definition 3 in the paper), computed per-block.
    Unaffected by RGB packing — this measures the DCT-cutoff compression,
    not ciphertext expansion.
    """
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


def print_rgb_bandwidth_summary(y_bytes: int, chroma_bytes: int, direction: str):
    """
    Report combined bandwidth and the saving vs. naive 3-channel encryption
    (three independent grayscale-style pipelines for R, G, B).
    """
    total = y_bytes + chroma_bytes
    naive_3x = 3 * y_bytes   # each channel would cost the same as one Y set
    print(f"\n  [{direction}] Y set        : {y_bytes:,} bytes")
    print(f"  [{direction}] Cr+Cb packed : {chroma_bytes:,} bytes")
    print(f"  [{direction}] TOTAL        : {total:,} bytes  "
          f"({total / (1024*1024):.3f} MB)")
    print(f"  [{direction}] vs. naive 3-channel encryption ({naive_3x:,} bytes): "
          f"{total / naive_3x:.2f}x  ({100 * (1 - total/naive_3x):.1f}% smaller)")


# =============================================================================
#  CLIENT — COMPRESS A SINGLE PLANE (shared by Y, Cr, Cb)
# =============================================================================

def compress_plane(plane: np.ndarray, m: int, c: int, quality: int,
                    use_overlap_trick: bool):
    """
    Block + level + DCT + quantize + zigzag + cutoff for ONE plane.
    Identical logic to the grayscale pipeline's compression step, just
    factored out so it can be reused for Y, Cr, and Cb independently.

    Returns
    -------
    compressed_payload : np.ndarray (N_blocks, c)
    orig_shape          : (H, W) of this plane before padding
    """
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


def encrypt_single_plane(compressed_payload: np.ndarray, context, c: int):
    """One ciphertext per frequency lane — the original grayscale scheme."""
    encrypted = []
    for fi in range(c):
        lane = compressed_payload[:, fi].tolist()
        with silence_stdout_stderr():
            encrypted.append(ts.ckks_vector(context, lane))
    return encrypted


def encrypt_packed_pair(payload_a: np.ndarray, payload_b: np.ndarray,
                         context, c: int):
    """
    Pack two planes' block data into the SAME ciphertext's SIMD slots,
    per frequency lane: [payload_a[:, fi] || payload_b[:, fi]].
    Still only `c` ciphertexts total for BOTH planes combined.
    """
    encrypted = []
    for fi in range(c):
        lane = payload_a[:, fi].tolist() + payload_b[:, fi].tolist()
        with silence_stdout_stderr():
            encrypted.append(ts.ckks_vector(context, lane))
    return encrypted


def client_compress_and_encrypt_rgb(bgr_image: np.ndarray, context, m: int,
                                     c: int, quality: int,
                                     use_overlap_trick: bool):
    """
    Full client-side RGB compression + encryption.

    Returns
    -------
    y_cts       : list[CKKSVector]  length c  (Y channel)
    chroma_cts  : list[CKKSVector]  length c  (Cr+Cb packed)
    51        : dict with everything needed to reconstruct the image
    """
    print("--- Client: Splitting color image into Y / Cr / Cb ---")
    y_plane, cr_plane, cb_plane = split_and_subsample_chroma(bgr_image)
    print(f"  [Client] Y  plane: {y_plane.shape}   "
          f"Cr/Cb planes (subsampled 4:2:0): {cr_plane.shape}")

    print("\n--- Client: Compressing Y ---")
    y_payload, y_shape = compress_plane(y_plane, m, c, quality, use_overlap_trick)
    print(f"  [Client] Y: {y_payload.shape[0]} blocks × {c} coefficients")

    print("--- Client: Compressing Cr, Cb ---")
    cr_payload, chroma_shape = compress_plane(cr_plane, m, c, quality, use_overlap_trick)
    cb_payload, _            = compress_plane(cb_plane, m, c, quality, use_overlap_trick)
    print(f"  [Client] Cr/Cb: {cr_payload.shape[0]} blocks each × {c} coefficients "
          f"(packed -> {2 * cr_payload.shape[0]} slots per ciphertext)")

    print("\n--- Client: Encrypting Y (c ciphertexts) ---")
    y_cts = encrypt_single_plane(y_payload, context, c)

    print("--- Client: Encrypting Cr+Cb packed (c ciphertexts) ---")
    chroma_cts = encrypt_packed_pair(cr_payload, cb_payload, context, c)

    meta = {
        'orig_hw':       bgr_image.shape[:2],
        'y_shape':       y_shape,
        'chroma_shape':  chroma_shape,
        'y_blocks':      y_payload.shape[0],
        'chroma_blocks': cr_payload.shape[0],   # Cr and Cb always match
    }
    return y_cts, chroma_cts, meta


# =============================================================================
#  SERVER — HOMOMORPHIC DECOMPRESS  (unchanged from grayscale pipeline)
# =============================================================================

def server_homomorphic_decompress(encrypted_cts, context, m: int, c: int,
                                   num_blocks: int, quality: int):
    """
    Two-pass separable IDCT over CKKS ciphertexts. Operates blindly on
    whatever the ciphertext's slots contain — works identically whether
    those slots hold one plane's blocks or two packed planes' blocks.
    """
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
#  SERVER — PROCESSING OPERATIONS  (unchanged from grayscale pipeline)
# =============================================================================

def server_process_pixelwise(decompressed_grid, m: int):
    """Apply pixel-wise brightening (×1.3) homomorphically."""
    print("\n--- Server: Applying pixel-wise brightening ---")
    for i in range(m):
        for j in range(m):
            if isinstance(decompressed_grid[i][j], ts.CKKSVector):
                decompressed_grid[i][j] *= 1.3
    return decompressed_grid


def server_process_convolution(decompressed_grid, m: int):
    """Apply a 3×3 sharpening kernel homomorphically."""
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


def server_process_channel(decompressed_grid, m: int, operation: str,
                            apply_to_this_channel: bool):
    """
    Dispatch helper: decides whether THIS channel group gets the operation
    applied, or passes through unchanged.

    apply_to_this_channel=False is used for the Cr/Cb (chroma) group under
    'Pixel' mode — brightening should adjust luma only, not shift color
    balance. Under 'Convolution' mode, both Y and chroma get sharpened.
    """
    if not apply_to_this_channel:
        return decompressed_grid   # pass-through: already valid CKKSVectors
    if operation == 'Pixel':
        return server_process_pixelwise(decompressed_grid, m)
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
    """
    Same reconstruction logic as the grayscale pipeline's decrypt step, but
    taking an already-decrypted (N_blocks, c) array directly — this lets it
    be reused for Y (straight decrypt) and for Cr/Cb (post-split from a
    packed decrypt).
    """
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


def client_decrypt_single_plane(encrypted_vectors, orig_shape, m: int, c: int,
                                 quality: int, use_overlap_trick: bool) -> np.ndarray:
    """Decrypt a plain (non-packed) ciphertext set — used for Y."""
    decrypted_lanes = np.array([ct.decrypt() for ct in encrypted_vectors]).T  # (N, c)
    return reconstruct_plane_from_lanes(decrypted_lanes, orig_shape, m, c,
                                         quality, use_overlap_trick)


def client_decrypt_packed_pair(encrypted_vectors, shape_a, shape_b,
                                blocks_a: int, m: int, c: int, quality: int,
                                use_overlap_trick: bool):
    """
    Decrypt c ciphertexts, each holding [plane_a_lane || plane_b_lane] in
    its slots, split by the known block-count boundary, and reconstruct
    both planes. Used for Cr+Cb.
    """
    decrypted = np.array([ct.decrypt() for ct in encrypted_vectors])  # (c, blocks_a+blocks_b)
    lanes_a = decrypted[:, :blocks_a].T   # (blocks_a, c)
    lanes_b = decrypted[:, blocks_a:].T   # (blocks_b, c)

    plane_a = reconstruct_plane_from_lanes(lanes_a, shape_a, m, c, quality, use_overlap_trick)
    plane_b = reconstruct_plane_from_lanes(lanes_b, shape_b, m, c, quality, use_overlap_trick)
    return plane_a, plane_b


# =============================================================================
#  SUMMARY PRINTING
# =============================================================================

def print_rgb_pipeline_summary(image_size, operation, m, c, use_overlap_trick,
                                y_original, y_reconstructed,
                                total_time, y_bytes_out, y_bytes_in,
                                chroma_bytes_out, chroma_bytes_in):
    ssi_val = compute_ssi(y_original, y_reconstructed)
    _, _, ratio_pct = compute_compression_ratio(m, c, use_overlap_trick)

    print("\n" + "=" * 60)
    print(f"  {image_size}x{image_size} RGB  |  {operation}  |  m={m}  |  c={c}")
    print("=" * 60)
    print(f"  SSI (Y channel)      : {ssi_val:.3f}")
    print(f"  Compression ratio    : 100:{ratio_pct:.1f}")
    print(f"  Total pipeline time  : {total_time:.2f} s")
    print_rgb_bandwidth_summary(y_bytes_out, chroma_bytes_out, "Client -> Server")
    print_rgb_bandwidth_summary(y_bytes_in, chroma_bytes_in, "Server -> Client")


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def main_execution_pipeline():
    """Interactive end-to-end FHE RGB image compression pipeline."""

    print("=" * 60)
    print("  FHE Convolution-Friendly Image Compression Pipeline (RGB)")
    print("=" * 60)
    print()

    # ── User inputs ───────────────────────────────────────────────────────
    image_size = int(input("Image size (256 / 512): "))
    if image_size not in (256, 512):
        raise ValueError("This RGB pipeline only supports 256 or 512.")

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

    # ── CKKS context ──────────────────────────────────────────────────────
    print("[Setup] Generating CKKS context …")
    context = generate_ckks_context(image_size)
    print("[Setup] Context ready.\n")

    pipeline_t0 = time.time()

    # ── Client: compress + encrypt (Y separate, Cr+Cb packed) ──────────────
    t0 = time.time()
    y_cts, chroma_cts, meta = client_compress_and_encrypt_rgb(
        bgr_image, context, m=macro_block, c=cutoff, quality=quality,
        use_overlap_trick=use_overlap,
    )
    print(f"\n  [Time] Compress + Encrypt : {time.time() - t0:.2f} s")

    print("\n[Bandwidth] Client -> Server:")
    y_bytes_out      = measure_bandwidth(y_cts, description="Y (c ciphertexts)")
    chroma_bytes_out = measure_bandwidth(chroma_cts, description="Cr+Cb packed (c ciphertexts)")
    print_rgb_bandwidth_summary(y_bytes_out, chroma_bytes_out, "Client -> Server")

    # ── Server: decompress -> process -> recompress, for BOTH groups ───────
    t0 = time.time()

    y_decompressed = server_homomorphic_decompress(
        y_cts, context, macro_block, cutoff, meta['y_blocks'], quality)
    y_processed = server_process_channel(
        y_decompressed, macro_block, operation, apply_to_this_channel=True)
    y_recompressed = server_homomorphic_compress(
        y_processed, macro_block, cutoff, context, quality)

    chroma_num_blocks = 2 * meta['chroma_blocks']
    chroma_decompressed = server_homomorphic_decompress(
        chroma_cts, context, macro_block, cutoff, chroma_num_blocks, quality)
    # Brightening: Y only (avoids color shift). Sharpening: both groups.
    chroma_processed = server_process_channel(
        chroma_decompressed, macro_block, operation,
        apply_to_this_channel=(operation == 'Convolution'))
    chroma_recompressed = server_homomorphic_compress(
        chroma_processed, macro_block, cutoff, context, quality)

    print(f"\n  [Time] Server Homomorphic Processing (Y + chroma) : {time.time() - t0:.2f} s")

    print("\n[Bandwidth] Server -> Client:")
    y_bytes_in      = measure_bandwidth(y_recompressed, description="Y (c ciphertexts)")
    chroma_bytes_in = measure_bandwidth(chroma_recompressed, description="Cr+Cb packed (c ciphertexts)")
    print_rgb_bandwidth_summary(y_bytes_in, chroma_bytes_in, "Server -> Client")

    # ── Client: decrypt + decompress + merge ────────────────────────────────
    t0 = time.time()
    y_result = client_decrypt_single_plane(
        y_recompressed, meta['y_shape'], macro_block, cutoff, quality, use_overlap)
    cr_result, cb_result = client_decrypt_packed_pair(
        chroma_recompressed, meta['chroma_shape'], meta['chroma_shape'],
        meta['chroma_blocks'], macro_block, cutoff, quality, use_overlap)
    result_bgr = merge_and_upsample_chroma(y_result, cr_result, cb_result)
    print(f"\n  [Time] Decrypt + Decompress + Merge : {time.time() - t0:.2f} s")

    total_time = time.time() - pipeline_t0

    # ── Reference Y plane for SSI (from the original, uncompressed image) ──
    y_original, _, _ = split_and_subsample_chroma(bgr_image)

    print_rgb_pipeline_summary(
        image_size, operation, macro_block, cutoff, use_overlap,
        y_original, y_result, total_time,
        y_bytes_out, y_bytes_in, chroma_bytes_out, chroma_bytes_in,
    )

    # ── Display ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"FHE RGB Pipeline — {image_size}×{image_size}  |  {operation}  |  "
        f"m={macro_block}  |  c={cutoff}",
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
#  [1] Use PNG source images, not JPEG — same reasoning as the grayscale
#      pipeline: JPEG's own DCT + 4:2:0 chroma subsampling would compound
#      with this pipeline's own compression and subsampling steps.
#
#  [2] Why brightening only touches Y
#      Multiplying Cr/Cb by a scalar would shift color balance, not just
#      brightness. If you want a "brighten equally in all channels" mode
#      instead, pass apply_to_this_channel=True for chroma too under
#      'Pixel' mode — but expect a visible color-saturation shift.
#
#  [3] Why 1024/2048 aren't supported here
#      See the capacity note in generate_ckks_context(). Supporting them
#      would need a deeper CKKS context (more RNS primes), which increases
#      per-ciphertext bandwidth — the same trade-off already discussed for
#      the grayscale pipeline, just revisited at the point where chroma
#      packing itself would also start needing more room.
#
#  [4] Further bandwidth reduction (not implemented here)
#      Chroma naturally carries less high-frequency energy than luma. A
#      smaller constant cutoff for chroma (e.g. c_chroma=10 instead of
#      c_luma=22) would reduce bandwidth further without any content-
#      dependent leakage (it's still a fixed, public constant — see the
#      paper's Section 3.2 argument). Left out here to keep this version's
#      logic straightforward; it requires the packed lanes to handle
#      c_chroma running out before c_luma does.
#
#  [5] Maximum-savings alternative (not implemented here)
#      Packing ALL THREE channels (Y+Cr+Cb) into one ciphertext set would
#      get bandwidth down to c total (matching grayscale exactly), but
#      requires a plaintext slot-mask multiply whenever an operation must
#      differ by channel (e.g. brightening Y only within a shared
#      ciphertext) — one extra CMult+CAdd, costing one more multiplicative
#      level. The 2c "Y separate" design here avoids that complexity.
# =============================================================================
