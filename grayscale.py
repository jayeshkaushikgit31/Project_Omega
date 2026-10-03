# =============================================================================
#  BTP — FHE Convolution-Friendly Image Compression Pipeline
#  Converted from Google Colab to PyCharm / local Python environment
# =============================================================================
#
#  SETUP (run once in your PyCharm terminal):
#      pip install tenseal numpy opencv-python scipy matplotlib
#
#  IMAGE FOLDER:
#      Create a folder called  images/  next to this script and place your
#      PNG test images inside it named as:  si_256_gray.png
#                                           si_512_gray.png
#                                           si_1024_gray.png
#                                           si_2048_gray.png
#      PNG is recommended over JPEG — see note at the bottom of this file.
#
#  HOW TO RUN IN PYCHARM:
#      Right-click → "Run btp_local.py"  or  Shift+F10
#      The terminal at the bottom will prompt for your choices interactively.
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

# Folder containing your test images (relative to this script's location).
IMAGES_DIR = Path(__file__).parent / "images"

# Supported image file extensions to try (in order of preference).
# PNG is lossless and preferred. JPEG is accepted but see note at bottom.
IMAGE_EXTENSIONS = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG",
                    ".bmp", ".BMP", ".tiff", ".TIFF"]

# =============================================================================
#  MATPLOTLIB BACKEND
#  PyCharm on Windows/Linux may need 'TkAgg'; on macOS 'MacOSX' usually works.
#  If plt.show() opens no window, change to: matplotlib.use('TkAgg')
# =============================================================================
matplotlib.use('TkAgg')   # <-- change to 'MacOSX' on macOS if needed

# =============================================================================
#  SUPPRESS TENSEAL / NUMPY WARNINGS
# =============================================================================

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
logging.getLogger('tenseal').setLevel(logging.ERROR)

# Cache dicts — avoids recomputing DCT / Q matrices on every call
_dct_cache   = {}
_quant_cache = {}


@contextmanager
def silence_stdout_stderr():
    """Redirect stdout and stderr to /dev/null (suppresses TenSEAL C++ output)."""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull), redirect_stdout(fnull):
            yield


# =============================================================================
#  IMAGE LOADING  (replaces the Google Drive version)
# =============================================================================

def load_image(image_size: int) -> np.ndarray:
    """
    Load a grayscale image from the local  images/  folder.

    Looks for:  images/si_{image_size}_gray.<ext>
    where <ext> is tried in the order defined by IMAGE_EXTENSIONS.

    Returns
    -------
    np.ndarray   shape (H, W), dtype uint8, values in [0, 255]
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    stem = f"si_{image_size}_gray"
    for ext in IMAGE_EXTENSIONS:
        candidate = IMAGES_DIR / f"{stem}{ext}"
        if candidate.exists():
            img = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"cv2 could not decode: {candidate}")
            print(f"[Image] Loaded '{candidate.name}'  shape={img.shape}  dtype={img.dtype}")
            return img

    # Nothing found — give a clear, actionable error
    raise FileNotFoundError(
        f"\n[Error] No image found for size {image_size}.\n"
        f"  Expected file : {IMAGES_DIR / stem}.<png|jpg|...>\n"
        f"  Images folder : {IMAGES_DIR.resolve()}\n"
        f"  Files present : {[f.name for f in IMAGES_DIR.iterdir()] if IMAGES_DIR.exists() else '(folder missing)'}\n"
        f"\n  Fix: place a grayscale PNG named  si_{image_size}_gray.png  in the images/ folder."
    )


# =============================================================================
#  CKKS CONTEXT
# =============================================================================

def generate_ckks_context(image_size: int) -> ts.Context:
    """
    Build a TenSEAL CKKS context sized for the given image dimension.

    Π1  (poly_modulus_degree=16384) — for 256×256 and 512×512
    Π2  (poly_modulus_degree=32768) — for 1024×1024 and 2048×2048

    Larger images produce more blocks; TenSEAL internally splits a
    CKKSVector across multiple ciphertext chunks when the block count
    exceeds poly_modulus_degree/2.  Raising the degree trades speed
    for capacity.  128-bit security is maintained for both settings.
    The 9-prime chain "coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60]",
    provides 7 multiplicative depth levels.
    """
    with silence_stdout_stderr():
        if image_size in (256, 512):
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=16384,
                coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60], 
            )
        elif image_size in (1024, 2048):
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=32768,
                coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 60],
            )
        else:
            raise ValueError(
                f"Unsupported image_size={image_size}. Choose 256, 512, 1024, or 2048."
            )
        context.global_scale = 2 ** 40

    return context


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================
def compute_ssi(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Structural Similarity Index (Definition 2 in the paper).
    Range [-1, 1]; 1 = perfect match. The paper considers ≥0.95 'very good'.
    Requires both images to be the same shape and dtype (uint8 here).
    """
    return ssim(original, reconstructed, data_range=255)


def compute_compression_ratio(m: int, c: int, use_overlap_trick: bool):
    """
    Compression ratio B0:B1 (Definition 3 in the paper), computed per-block.

    B0 = 'useful' pixels encoded by one block:
         - Pixel-wise      -> m²        (every pixel is new data)
         - Convolution     -> (m-2)²    (the 1px border is padding shared
                                          with neighbours, not new info)
    B1 = c  (coefficients actually kept per block)

    Returns (B0, B1, ratio_as_percent) so you can print "100:{ratio_as_percent:.1f}"
    exactly like Table 2 in the paper.
    """
    B0 = (m - 2) ** 2 if use_overlap_trick else m ** 2
    B1 = c
    return B0, B1, (B1 / B0) * 100


def print_paper_style_summary(image_size, setting_label, decompress_t,
                               process_t, compress_t, original, reconstructed,
                               m, c, use_overlap_trick):
    """Print a row matching the paper's Table 2 layout."""
    total_t = decompress_t + process_t + compress_t
    ssi_val = compute_ssi(original, reconstructed)
    _, _, ratio_pct = compute_compression_ratio(m, c, use_overlap_trick)

    print(f"\n{'Image Size':<12}{'Setting':<10}{'Decomp':<10}{'Process':<10}"
          f"{'Compress':<10}{'Total':<10}{'SSI':<8}{'Ratio':<12}")
    print(f"{image_size}×{image_size:<8}{setting_label:<10}"
          f"{decompress_t:<10.2f}{process_t:<10.3f}{compress_t:<10.2f}"
          f"{total_t:<10.2f}{ssi_val:<8.3f}100:{ratio_pct:.1f}")

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


# =============================================================================
#  CLIENT — COMPRESS AND ENCRYPT
# =============================================================================

def client_compress_and_encrypt(image: np.ndarray, context, m: int, c: int,
                                 quality: int, use_overlap_trick: bool):
    """
    Client-side compression and CKKS encryption.

    Steps
    -----
    1. Pad image so it divides evenly into m×m blocks.
       With the overlap trick, blocks are (m-2)×(m-2) with a 1-pixel
       border of neighbours — producing m×m blocks after padding.
    2. For each block: level → 2D-DCT → quantise → zigzag → cutoff(c).
    3. Transpose the (N_blocks × c) matrix so that all blocks' k-th
       frequency coefficient sit in one lane, then encrypt each lane
       into one CKKS ciphertext.  This gives c ciphertexts in total.

    Returns
    -------
    encrypted_simd_vectors  : list[CKKSVector]  length c
    (original_h, original_w): tuple[int, int]   for reconstruction
    total_blocks            : tuple              shape of block array
    """
    print("--- Client: Compressing and Encrypting image ---\n")
    original_h, original_w = image.shape
    Q    = get_quantization_matrix(m, quality)
    step = (m - 2) if use_overlap_trick else m

    pad_h = (step - original_h % step) % step
    pad_w = (step - original_w % step) % step
    if use_overlap_trick:
        pad_h += 2
        pad_w += 2

    working_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    padded_h, padded_w = working_image.shape

    compressed_payload = []
    for i in range(0, padded_h - (m - step), step):
        for j in range(0, padded_w - (m - step), step):
            block = working_image[i:i+m, j:j+m].astype(np.float64)
            leveled        = block - 128.0
            dct_freqs      = dct(dct(leveled, axis=0, norm='ortho'), axis=1, norm='ortho')
            quantized      = np.round(dct_freqs / Q)
            linear_vector  = execute_zigzag_scan(quantized, m)
            compressed_payload.append(linear_vector[:c])

    compressed_payload = np.array(compressed_payload)   # (N_blocks, c)
    total_blocks       = compressed_payload.shape
    print(f"  [Client] Compression yield: {total_blocks[0]} blocks × {c} coefficients")

    encrypted_simd_vectors = []
    print(f"  [Client] Encrypting {c} SIMD lanes …")
    for fi in range(c):
        lane = compressed_payload[:, fi].tolist()
        with silence_stdout_stderr():
            encrypted_simd_vectors.append(ts.ckks_vector(context, lane))

    return encrypted_simd_vectors, (original_h, original_w), total_blocks


# =============================================================================
#  SERVER — HOMOMORPHIC DECOMPRESS
# =============================================================================

def server_homomorphic_decompress(encrypted_cts, context, m: int, c: int,
                                   num_blocks: int, quality: int):
    """
    Two-pass separable IDCT over CKKS ciphertexts.

    Factors  E = Tᵀ @ D @ T  into:
      Pass 1 (sparse): intermediate[k][j] = Σ_l  D[k][l] · T[l][j]   — c·m ops
      Pass 2 (dense):  final[i][j]        = Σ_k  Tᵀ[i][k] · inter[k][j] — ≤m³ ops

    vs the naive quadruple loop: c·m² ops, m⁴ Python iterations.
    """
    print("\n--- Server: Homomorphically Decompressing ---")
    Q       = get_quantization_matrix(m, quality)
    T       = get_dct_matrix(m)
    T_T     = T.T
    zz_idx  = zigzag_indices(m)
    T_list  = T.tolist()
    TT_list = T_T.tolist()

    # Step 0 — inverse zigzag + de-quantise
    grid = [[None] * m for _ in range(m)]
    for idx in range(min(len(encrypted_cts), c)):
        r, col = zz_idx[idx]
        grid[r][col] = encrypted_cts[idx] * float(Q[r, col])

    # Pass 1 — row transform: intermediate = D @ T
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

    # Pass 2 — column transform: output = Tᵀ @ intermediate
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

    # Unlevel (+128)
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

    # Border pixels: pass through unchanged
    for i in range(m):
        for j in range(m):
            if processed_grid[i][j] is None:
                processed_grid[i][j] = decompressed_grid[i][j] * 1.0

    return processed_grid


# =============================================================================
#  SERVER — HOMOMORPHIC RECOMPRESS
# =============================================================================

def server_homomorphic_compress(processed_grid, m: int, c: int, context, quality: int):
    """
    Two-pass separable DCT over CKKS ciphertexts.

    Factors  B = T @ A @ Tᵀ  into:
      Pass 1 (dense):  intermediate[i][v] = Σ_j  A[i][j] · Tᵀ[j][v]  — m³ ops
      Pass 2 (sparse): B[u][v]            = Σ_i  T[u][i] · inter[i][v] — c·m ops
    """
    print("\n--- Server: Homomorphically Re-compressing ---")
    num_blocks = processed_grid[0][0].size()
    Q       = get_quantization_matrix(m, quality)
    T       = get_dct_matrix(m)
    T_T     = T.T
    zz_idx  = zigzag_indices(m)
    T_list  = T.tolist()
    TT_list = T_T.tolist()

    # Level (−128)
    for i in range(m):
        for j in range(m):
            if isinstance(processed_grid[i][j], ts.CKKSVector):
                processed_grid[i][j] -= 128.0

    # Pass 1 — intermediate = A @ Tᵀ
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

    # Pass 2 — only the c zigzag outputs needed
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

def client_decrypt_and_decompress(encrypted_vectors, context, orig_shape,
                                   m: int, c: int, quality: int,
                                   use_overlap_trick: bool) -> np.ndarray:
    """
    Decrypt server ciphertexts and reconstruct the spatial image.

    Steps
    -----
    1. Decrypt each ciphertext → one lane of (N_blocks,) floats.
    2. Stack and transpose → (N_blocks, c) block matrix.
    3. For each block: zero-pad to m², inverse-zigzag, de-quantise,
       2D-IDCT, unlevel, stitch into the output canvas.
    """
    print("\n--- Client: Decrypting and Decompressing image ---\n")
    original_h, original_w = orig_shape
    step  = (m - 2) if use_overlap_trick else m

    pad_h = (step - original_h % step) % step
    pad_w = (step - original_w % step) % step
    if use_overlap_trick:
        pad_h += 2
        pad_w += 2

    padded_h = original_h + pad_h
    padded_w = original_w + pad_w
    Q        = get_quantization_matrix(m, quality)

    decrypted_lanes  = np.array([ct.decrypt() for ct in encrypted_vectors]).T  # (N, c)
    reconstructed    = np.zeros((padded_h, padded_w), dtype=np.float64)

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


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def main_execution_pipeline():
    """Interactive end-to-end FHE image compression pipeline."""

    print("=" * 60)
    print("  FHE Convolution-Friendly Image Compression Pipeline")
    print("=" * 60)
    print()

    # ── User inputs ───────────────────────────────────────────────────────
    image_size = int(input("Image size (256 / 512 / 1024 / 2048): "))

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
        "Scaling quality q [1–99]  "
        "(50=identity | >50=preserve quality | <50=more compression): "
    ))

    print()

    # ── Load image ────────────────────────────────────────────────────────
    gray_image = load_image(image_size)         # uint8 (H, W), correct grayscale

    # ── CKKS context ──────────────────────────────────────────────────────
    print("[Setup] Generating CKKS context …")
    context = generate_ckks_context(image_size)
    print("[Setup] Context ready.\n")

    # ── Client: compress + encrypt ────────────────────────────────────────
    t0 = time.time()
    encrypted_payload, shape_meta, block_shape = client_compress_and_encrypt(
        gray_image, context,
        m=macro_block, c=cutoff, quality=quality, use_overlap_trick=use_overlap,
    )
    num_blocks = block_shape[0]
    print(f"\n  [Time] Compress + Encrypt : {time.time() - t0:.2f} s")

    print("\n[Bandwidth] Client → Server:")
    measure_bandwidth(encrypted_payload, description="Encrypted payload")

    # ── Server: decompress → process → recompress ─────────────────────────


    # In main_execution_pipeline(), split the timing:
    t0 = time.time()
    decompressed = server_homomorphic_decompress(
        encrypted_payload, context, macro_block, cutoff, num_blocks, quality)
    t_decompress = time.time() - t0

    t0 = time.time()
    processed = server_process_pixelwise(decompressed, m=macro_block) if operation == 'Pixel' \
                else server_process_convolution(decompressed, m=macro_block)
    t_process = time.time() - t0

    t0 = time.time()
    recompressed = server_homomorphic_compress(
        processed, m=macro_block, c=cutoff, context=context, quality=quality)
    t_compress = time.time() - t0

    print("\n[Bandwidth] Server → Client:")
    measure_bandwidth(recompressed, description="Processed payload")

    # ── Client: decrypt + decompress ──────────────────────────────────────
    t0 = time.time()
    result = client_decrypt_and_decompress(
        recompressed, context, shape_meta,
        m=macro_block, c=cutoff, quality=quality, use_overlap_trick=use_overlap,
    )
    print(f"\n  [Time] Decrypt + Decompress : {time.time() - t0:.2f} s")

    print_paper_style_summary(
        image_size, operation, t_decompress, t_process, t_compress,
        gray_image, result, macro_block, cutoff, use_overlap,
    )

    # ── Display ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"FHE Pipeline — {image_size}×{image_size}  |  {operation}  |  m={macro_block}  |  c={cutoff}",
        fontsize=12,
    )
    axes[0].imshow(gray_image, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(result, cmap='gray')
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
#  [1] Use PNG source images, not JPEG
#      Loading a JPEG source introduces double compression: JPEG's own DCT
#      lossy encoding is applied before your pipeline adds a second round.
#      At cutoff c=22 this is masked (~0.001 SSI difference), but for higher
#      fidelity settings (c≥40) the gap widens.  More importantly, JPEG
#      performs 4:2:0 chroma subsampling by default, which compounds with
#      any chroma processing you add.  Use lossless PNG source images —
#      cv2.imread handles them identically, just change the filename.
#
#  [2] Why q barely affects bandwidth
#      In standard JPEG, lower quality → more zeros → smaller file after
#      entropy coding.  Here every frequency coefficient (even a zero) is
#      packed into its own CKKS ciphertext slot, so the ciphertext count
#      is fixed at c regardless of how many values are zero.  To reduce
#      bandwidth, lower c — not q.  The q parameter only affects the
#      mathematical rounding of values inside the slots.
#
#  [3] Variable cutoff and information leakage
#      Using a different cutoff per block (like standard JPEG's entropy
#      encoding) would leak structural information: an attacker who sees
#      that block A compressed to 10 ciphertexts and block B to 40 knows
#      A is smooth (background) and B is textured (foreground), allowing
#      silhouette reconstruction without decrypting anything.  The constant
#      cutoff c is what prevents this side-channel.
#
#  [4] CKKS ciphertext expansion
#      Ciphertext expansion still occurs (plaintext bytes → megabytes per
#      ciphertext), but the compression step ensures fewer ciphertexts are
#      sent in the first place, keeping total bandwidth lower than the
#      no-compression baseline.
# =============================================================================
