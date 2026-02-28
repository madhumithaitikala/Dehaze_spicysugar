"""
add_haze.py - Artificially add haze to clear images using the Atmospheric Scattering Model.

The standard physical model for haze is:
    I(x) = J(x) * t(x) + A * (1 - t(x))

Where:
    I(x) = observed hazy image
    J(x) = clear scene radiance (input clear image)
    t(x) = transmission map  (controls haze density)
    A    = global atmospheric light (color of the haze)

This script supports multiple haze modes:
    - uniform:  constant transmission (uniform fog)
    - depth:    simulated depth-based haze (objects further away get hazier)
    - random:   random patchy fog with Perlin-like noise

Usage:
    python add_haze.py --input_dir data/I-HAZE/test/clear --output_dir results/synthetic_hazy --mode uniform --intensity 0.7
    python add_haze.py --input path/to/single_image.png --output path/to/hazy_image.png --mode depth --intensity 0.5
"""

import argparse
import os
import glob
import cv2
import numpy as np


def generate_transmission_map(shape, mode='uniform', intensity=0.5):
    """
    Generate a transmission map t(x) for the atmospheric scattering model.

    Args:
        shape: (H, W) tuple
        mode: 'uniform', 'depth', or 'random'
        intensity: float in [0, 1]. Higher = more haze.
                   For uniform: t = 1 - intensity
                   For depth: controls the beta*d falloff
    Returns:
        t: transmission map of shape (H, W), values in [0, 1]
    """
    h, w = shape

    if mode == 'uniform':
        # Constant transmission across the entire image
        t = np.full((h, w), 1.0 - intensity, dtype=np.float64)

    elif mode == 'depth':
        # Simulate depth: top of image = far (more haze), bottom = near (less haze)
        # This mimics outdoor scenes where sky/horizon is further away
        beta = intensity * 3.0  # scattering coefficient
        # Create a linear depth map (top=far, bottom=near)
        depth = np.linspace(1.0, 0.1, h).reshape(h, 1)
        depth = np.tile(depth, (1, w))
        # Add some horizontal variation
        x_var = np.linspace(0.8, 1.2, w).reshape(1, w)
        x_var = np.tile(x_var, (h, 1))
        depth = depth * x_var
        # t(x) = exp(-beta * d(x))
        t = np.exp(-beta * depth)
        t = np.clip(t, 0.05, 1.0)

    elif mode == 'random':
        # Patchy fog using multi-scale Gaussian noise
        t_base = 1.0 - intensity
        # Create smooth random noise at multiple scales
        noise = np.zeros((h, w), dtype=np.float64)
        for scale in [8, 16, 32, 64]:
            small = np.random.randn(max(1, h // scale), max(1, w // scale))
            upscaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
            noise += upscaled * (1.0 / scale)
        # Normalize noise to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        t = t_base + noise * intensity * 0.5
        t = np.clip(t, 0.05, 1.0)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'uniform', 'depth', or 'random'.")

    return t


def add_haze(clear_image, mode='uniform', intensity=0.5, atmospheric_light=(0.8, 0.85, 0.9)):
    """
    Apply the Atmospheric Scattering Model to a clear image.

    Args:
        clear_image: numpy array (H, W, 3) in BGR, uint8
        mode: haze type ('uniform', 'depth', 'random')
        intensity: float in [0, 1], controls haze density
        atmospheric_light: tuple (B, G, R) normalized atmospheric light color

    Returns:
        hazy_image: numpy array (H, W, 3) in BGR, uint8
    """
    h, w = clear_image.shape[:2]

    # Normalize image to [0, 1]
    J = clear_image.astype(np.float64) / 255.0

    # Generate transmission map
    t = generate_transmission_map((h, w), mode=mode, intensity=intensity)

    # Expand dims for broadcasting: (H, W) -> (H, W, 1)
    t_3d = t[:, :, np.newaxis]

    # Atmospheric light as array (BGR)
    A = np.array(atmospheric_light, dtype=np.float64).reshape(1, 1, 3)

    # Apply atmospheric scattering model: I = J * t + A * (1 - t)
    I = J * t_3d + A * (1.0 - t_3d)

    # Clip and convert back to uint8
    I = np.clip(I * 255.0, 0, 255).astype(np.uint8)
    return I


def process_single(input_path, output_path, mode, intensity):
    """Process a single image."""
    img = cv2.imread(input_path)
    if img is None:
        print(f"  [ERROR] Could not read {input_path}")
        return
    hazy = add_haze(img, mode=mode, intensity=intensity)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    cv2.imwrite(output_path, hazy)
    print(f"  {os.path.basename(input_path)} -> {output_path}")


def process_directory(input_dir, output_dir, mode, intensity):
    """Process all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(input_dir, '*.*')))

    count = 0
    for img_path in images:
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue
        out_path = os.path.join(output_dir, os.path.basename(img_path))
        process_single(img_path, out_path, mode, intensity)
        count += 1

    print(f"\n  Processed {count} images with mode='{mode}', intensity={intensity}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Add synthetic haze to clear images using the Atmospheric Scattering Model."
    )
    parser.add_argument('--input', default=None, help="Path to a single clear image.")
    parser.add_argument('--output', default=None, help="Output path for the single hazy image.")
    parser.add_argument('--input_dir', default=None, help="Directory of clear images.")
    parser.add_argument('--output_dir', default=None, help="Directory to save hazy images.")
    parser.add_argument('--mode', default='uniform', choices=['uniform', 'depth', 'random'],
                        help="Haze mode: uniform, depth, or random. Default: uniform")
    parser.add_argument('--intensity', type=float, default=0.5,
                        help="Haze intensity [0.0 - 1.0]. Higher = more haze. Default: 0.5")
    args = parser.parse_args()

    print(f"\n  Haze Synthesis Pipeline")
    print(f"  Model: Atmospheric Scattering (Koschmieder's Law)")
    print(f"  I(x) = J(x) * t(x) + A * (1 - t(x))")
    print(f"  Mode: {args.mode} | Intensity: {args.intensity}\n")

    if args.input and args.output:
        process_single(args.input, args.output, args.mode, args.intensity)
    elif args.input_dir and args.output_dir:
        process_directory(args.input_dir, args.output_dir, args.mode, args.intensity)
    else:
        print("  Error: Provide either --input/--output or --input_dir/--output_dir")
