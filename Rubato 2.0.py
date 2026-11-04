#!/usr/bin/env python3
"""
Simulation of Rubato → CKKS transciphering for RGB images.
Optimizations: 11x11 tiles for convolution (prevents TenSEAL chunking), packed RGB channels.
Rubato layer is simulated (reversible XOR keystream, byte-accurate size).
CKKS operations are real (TenSEAL).
"""

import os
import time
import numpy as np
from PIL import Image
import tenseal as ts

# ----------------------------------------------------------------------
# Simulated Rubato parameters
# ----------------------------------------------------------------------
RUBATO_OVERHEAD = 16
RUBATO_KEY = b"simulated_rubato_key_32bytes!!!!"
RUBATO_DECRYPT_TIME = 0.0001


def simulate_rubato_encrypt(plaintext_bytes: bytes) -> bytes:
    n = len(plaintext_bytes)
    ks = (RUBATO_KEY * (n // len(RUBATO_KEY) + 1))[:n]
    ct_body = bytes(p ^ k for p, k in zip(plaintext_bytes, ks))
    return os.urandom(RUBATO_OVERHEAD) + ct_body


def simulate_rubato_decrypt(ciphertext_bytes: bytes) -> bytes:
    ct_body = ciphertext_bytes[RUBATO_OVERHEAD:]
    n = len(ct_body)
    ks = (RUBATO_KEY * (n // len(RUBATO_KEY) + 1))[:n]
    return bytes(c ^ k for c, k in zip(ct_body, ks))


# ----------------------------------------------------------------------
# CKKS context
# ----------------------------------------------------------------------
def create_ckks_context(n_slots_needed: int):
    N = 1
    while N // 2 <= n_slots_needed:
        N <<= 1

    if N == 8192:
        coeff = [60, 40, 40, 60]
    elif N == 16384:
        coeff = [60, 40, 40, 40, 40, 40, 40, 38]
    elif N == 32768:
        coeff = [60] + [40] * 10 + [38]
    else:
        raise ValueError(f"N={N} unsupported. Reduce tile size.")

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=N,
        coeff_mod_bit_sizes=coeff,
    )
    ctx.global_scale = 2 ** 40
    ctx.generate_galois_keys()
    return ctx, N


# ----------------------------------------------------------------------
# Tile extraction
# ----------------------------------------------------------------------
def extract_tiles_pixel(img, tile_size):
    H, W, C = img.shape
    tiles = []
    for r in range(0, H, tile_size):
        for c in range(0, W, tile_size):
            tile = img[r:r + tile_size, c:c + tile_size, :]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile = np.pad(
                    tile,
                    ((0, tile_size - tile.shape[0]),
                     (0, tile_size - tile.shape[1]),
                     (0, 0)),
                    mode="constant",
                )
            tiles.append((tile, r, c))
    return tiles, (H, W, C)


def extract_tiles_conv(img, tile_size):
    H, W, C = img.shape
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="constant")
    ext = tile_size + 2
    tiles = []

    for r in range(0, H, tile_size):
        for c in range(0, W, tile_size):
            tile = padded[r:r + ext, c:c + ext, :]
            if tile.shape[0] < ext or tile.shape[1] < ext:
                tile = np.pad(
                    tile,
                    ((0, ext - tile.shape[0]),
                     (0, ext - tile.shape[1]),
                     (0, 0)),
                    mode="constant",
                )
            tiles.append((tile, r, c))

    return tiles, (H, W, C)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    img_size = int(input("Enter image size (256, 512, 1024, 2048): ").strip())
    op = input("Enter operation (Pixel or Convolution): ").strip().lower()

    if op not in ("pixel", "convolution"):
        raise ValueError("Operation must be 'Pixel' or 'Convolution'")

    img_path = f"images/si_{img_size}_rgb.png"
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    original_img = Image.open(img_path).convert("RGB")
    img_array = np.array(original_img, dtype=np.float64)

    # ---- tile size and CKKS slot requirement ------------------------
    if op == "pixel":
        TILE_SIZE = 64
        tiles, out_shape = extract_tiles_pixel(img_array, TILE_SIZE)
        n_slots_needed = TILE_SIZE * TILE_SIZE * 3
    else:
        TILE_SIZE = 11
        tiles, out_shape = extract_tiles_conv(img_array, TILE_SIZE)
        # ext=13, width=3*13+4=43. windows = 11*41 = 451.
        # TenSEAL pads windows to 512. slots = 512 * 9 = 4608.
        n_slots_needed = 4608

    ctx, N = create_ckks_context(n_slots_needed)
    print(f"CKKS: N={N}, slots={N // 2}, tile={TILE_SIZE}, op={op}")
    print(f"Tiles: {len(tiles)}")

    # ---- Client -> Server: simulated Rubato encryption ---------------
    ciphertexts = []
    bw_c2s = 0

    for tile, r, c in tiles:
        if op == "pixel":
            data = tile.flatten().astype(np.float64)
        else:
            # Pack channels horizontally: C1 | 0 | C2 | 0 | C3
            pad_w = np.zeros((tile.shape[0], 2))
            packed_2d = np.concatenate([
                tile[:, :, 0], pad_w,
                tile[:, :, 1], pad_w,
                tile[:, :, 2]
            ], axis=1)  # shape (13, 43)
            data = packed_2d.flatten().astype(np.float64)

        ct = simulate_rubato_encrypt(data.tobytes())
        ciphertexts.append((ct, r, c, tile.shape[0], tile.shape[1]))
        bw_c2s += len(ct)

    # ---- Server processing ------------------------------------------
    server_start = time.perf_counter()
    processed = []
    bw_s2c = 0

    kernel = np.array(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]],
        dtype=np.float64,
    )

    n_total = len(ciphertexts)

    for idx, (ct, r, c, tH, tW) in enumerate(ciphertexts):
        if idx % 50 == 0:
            print(f"  server: {idx}/{n_total}...", end="\r")

        time.sleep(RUBATO_DECRYPT_TIME)
        pt_bytes = simulate_rubato_decrypt(ct)
        data = np.frombuffer(pt_bytes, dtype=np.float64).copy()

        if op == "pixel":
            enc = ts.ckks_vector(ctx, data)
            res_enc = enc * (-1) + 255
            res_plain = np.array(res_enc.decrypt(), dtype=np.float64)
            outH, outW = 64, 64
        else:
            packed_2d = data.reshape(tH, 43)
            enc, windows_nb = ts.im2col_encoding(
                ctx,
                packed_2d.tolist(),
                kernel.shape[0],
                kernel.shape[1],
                1,
            )

            res_enc = enc.conv2d_im2col(kernel.tolist(), windows_nb)
            res_plain = np.array(res_enc.decrypt(), dtype=np.float64)

            if len(res_plain) >= windows_nb:
                res_plain = res_plain[:windows_nb]

            # Output is (11, 41)
            res_2d = res_plain.reshape(11, 41)

            c1 = res_2d[:, 0:11]
            c2 = res_2d[:, 13:24]
            c3 = res_2d[:, 26:37]

            out_tile = np.stack([c1, c2, c3], axis=-1)
            res_plain = out_tile.flatten()
            outH, outW = 11, 11

        res_ct = simulate_rubato_encrypt(res_plain.tobytes())
        processed.append((res_ct, r, c, outH, outW))
        bw_s2c += len(res_ct)

    server_time = time.perf_counter() - server_start
    print("  server: complete.                  ")

    # ---- Client reconstruction --------------------------------------
    out = np.zeros(out_shape, dtype=np.float64)

    for ct, r, c, oH, oW in processed:
        pt_bytes = simulate_rubato_decrypt(ct)
        data = np.frombuffer(pt_bytes, dtype=np.float64).copy()
        t = data.reshape(oH, oW, 3)

        rh = min(oH, out_shape[0] - r)
        rw = min(oW, out_shape[1] - c)
        out[r:r + rh, c:c + rw, :] = t[:rh, :rw, :]

    out = np.clip(out, 0, 255).astype(np.uint8)

    # ---- Report ------------------------------------------------------
    print("\n========== REPORT ==========")
    print(f"Image:              {img_size}x{img_size} RGB")
    print(f"Operation:          {op}")
    print(f"CKKS N:             {N} (slots = {N // 2})")
    print(f"Tile size:          {TILE_SIZE}x{TILE_SIZE}")
    print(f"Number of tiles:    {len(tiles)}")
    print(f"Bandwidth C->S:     {bw_c2s / (1024 * 1024):.2f} MB")
    print(f"Bandwidth S->C:     {bw_s2c / (1024 * 1024):.2f} MB")
    print(f"Server time:        {server_time:.3f} s")
    print("============================\n")

    # ---- Side-by-side display ---------------------------------------
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_img)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(out)
        axes[1].set_title(f"Processed ({op})")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    except ImportError:
        Image.fromarray(out).save(f"result_{img_size}_{op}.png")
        print(
            f"matplotlib not installed. "
            f"Result saved to result_{img_size}_{op}.png"
        )


if __name__ == "__main__":
    main()