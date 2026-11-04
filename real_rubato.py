#!/usr/bin/env python3
import os
import json
import subprocess
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GO_BINARY_PATH = os.path.join(BASE_DIR, "go_real_rubato", "rubato_transcipher.exe")


def extract_tiles(img, tile_size):
    H, W, C = img.shape
    padded_img = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    ext_size = tile_size + 2

    tiles = []
    coords = []
    for r in range(0, H, tile_size):
        for c in range(0, W, tile_size):
            tile = padded_img[r:r + ext_size, c:c + ext_size, :]
            if tile.shape[0] < ext_size or tile.shape[1] < ext_size:
                tile = np.pad(tile, ((0, ext_size - tile.shape[0]),
                                     (0, ext_size - tile.shape[1]),
                                     (0, 0)), mode="edge")
            tiles.append(tile.flatten().tolist())
            coords.append((r, c))
    return tiles, coords, (H, W, C)


def main():
    if not os.path.exists(GO_BINARY_PATH):
        print(f"ERROR: Go binary not found at {GO_BINARY_PATH}")
        return

    img_size = int(input("Enter image size (256, 512, 1024, 2048): ").strip())
    op = input("Enter operation (Pixel or Convolution): ").strip().lower()

    img_path = f"images/si_{img_size}_rgb.png"
    original_img = Image.open(img_path).convert("RGB")
    img_array = np.array(original_img, dtype=np.float64)

    TILE_SIZE = 32
    tiles, coords, out_shape = extract_tiles(img_array, TILE_SIZE)

    print(f"Sending {len(tiles)} tiles to Go server for batch processing...")
    print("NOTE: Initializing real MT-PRO cryptographic keys and performing 8 rotations per tile...")

    request = {
        "operation": op,
        "tiles": tiles,
        "tile_size": TILE_SIZE
    }

    start = time.time()
    result = subprocess.run(
        [GO_BINARY_PATH],
        input=json.dumps(request),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"\nGo error:\n{result.stderr}")
        return

    response = json.loads(result.stdout)

    out = np.zeros(out_shape, dtype=np.float64)
    for i, (r, c) in enumerate(coords):
        out_tile = np.array(response["tiles"][i], dtype=np.float64).reshape(TILE_SIZE, TILE_SIZE, 3)
        rh = min(TILE_SIZE, out_shape[0] - r)
        rw = min(TILE_SIZE, out_shape[1] - c)
        out[r:r + rh, c:c + rw, :] = out_tile[:rh, :rw, :]

    out = np.clip(out, 0, 255).astype(np.uint8)

    print("\n========== REAL-WORLD REPORT ==========")
    print(f"Image:              {img_size}x{img_size} RGB")
    print(f"Operation:          {op}")
    print(f"Bandwidth C->S:     {response['bandwidth'] / (1024 * 1024):.2f} MB")
    print(f"Go Server Crypto Time: {response['server_time']:.4f} s")
    print(f"Python Total Roundtrip: {time.time() - start:.2f} s")
    print("=======================================\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(out)
    axes[1].set_title(f"Processed ({op})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()