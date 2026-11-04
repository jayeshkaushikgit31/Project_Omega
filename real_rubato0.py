#!/usr/bin/env python3
"""
Real-world Hybrid Go-Python Rubato Transciphering Pipeline.
"""

import os
import time
import json
import subprocess
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GO_BINARY_PATH = os.path.join(BASE_DIR, "", "go_rubato_server/rubato_transcipher.exe")


def extract_tiles(img, tile_size):
    """Splits an RGB image into non-overlapping tiles."""
    H, W, C = img.shape
    tiles = []
    for r in range(0, H, tile_size):
        for c in range(0, W, tile_size):
            tile = img[r:r + tile_size, c:c + tile_size, :]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile = np.pad(tile, ((0, tile_size - tile.shape[0]),
                                     (0, tile_size - tile.shape[1]),
                                     (0, 0)), mode="constant")
            tiles.append((tile, r, c))
    return tiles, (H, W, C)


def main():
    # 1. Check if Go binary exists
    if not os.path.exists(GO_BINARY_PATH):
        print(f"❌ ERROR: Cannot find Go binary at:\n   {GO_BINARY_PATH}")
        print("Please make sure you ran 'go build -o rubato_transcipher.exe .' inside the go_rubato_server folder.")
        return

    # 2. User Input
    img_size = int(input("Enter image size (256, 512, 1024, 2048): ").strip())
    op = input("Enter operation (Pixel or Convolution): ").strip().lower()

    if op not in ("pixel", "convolution"):
        raise ValueError("Operation must be 'Pixel' or 'Convolution'")

    # 3. Load Image
    img_path = f"images/si_{img_size}_rgb.png"
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    original_img = Image.open(img_path).convert("RGB")
    img_array = np.array(original_img, dtype=np.float64)

    # 4. Tiling
    TILE_SIZE = 32
    tiles, out_shape = extract_tiles(img_array, TILE_SIZE)

    print(f"Processing {len(tiles)} tiles of size {TILE_SIZE}x{TILE_SIZE}...")

    # 5. Client -> Server (Calling Go Binary)
    processed_tiles = []
    total_bandwidth = 0
    total_server_time = 0.0

    for idx, (tile, r, c) in enumerate(tiles):
        flat_data = tile.flatten().tolist()

        request = {
            "operation": op,
            "data": flat_data,
            "tile_size": TILE_SIZE
        }

        # Call the Go binary
        result = subprocess.run(
            [GO_BINARY_PATH],
            input=json.dumps(request),  # <--- FIX: Removed .encode('utf-8')
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"\nGo error on tile {idx}: {result.stderr}")
            continue

        response = json.loads(result.stdout)

        out_flat = np.array(response["data"], dtype=np.float64)
        out_tile = out_flat.reshape(TILE_SIZE, TILE_SIZE, 3)

        total_bandwidth += response["bandwidth"]
        total_server_time += response["server_time"]

        processed_tiles.append((out_tile, r, c))

        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(tiles)} tiles...", end="\r")

    print("  All tiles processed.                    ")

    # 6. Client Reconstruction
    out = np.zeros(out_shape, dtype=np.float64)
    for tile, r, c in processed_tiles:
        rh = min(TILE_SIZE, out_shape[0] - r)
        rw = min(TILE_SIZE, out_shape[1] - c)
        out[r:r + rh, c:c + rw, :] = tile[:rh, :rw, :]

    out = np.clip(out, 0, 255).astype(np.uint8)

    # 7. Report
    print("\n========== REAL-WORLD REPORT (Go-Python) ==========")
    print(f"Image:              {img_size}x{img_size} RGB")
    print(f"Operation:          {op}")
    print(f"Bandwidth C->S:     {total_bandwidth / (1024 * 1024):.2f} MB")
    print(f"Server time:        {total_server_time:.3f} s")
    print("====================================================\n")

    # 8. Display
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