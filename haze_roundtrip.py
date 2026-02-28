"""
haze_roundtrip.py - Complete round-trip pipeline:
    Clear Image → Add Synthetic Haze → Dehaze with Model → Evaluate PSNR/SSIM

This demonstrates the full cycle:
    1. Takes clear ground truth images
    2. Adds synthetic haze using the atmospheric scattering model
    3. Dehazes the synthetically hazed images using Cycle-Dehaze
    4. Computes PSNR/SSIM between the recovered image and the original clear image

Usage:
    python haze_roundtrip.py --clear_dir data/I-HAZE/test/clear --model models/Hazy2GT_indoor.pb --mode uniform --intensity 0.6
"""

import argparse
import os
import glob
import cv2
import re
import numpy as np
import tensorflow as tf
from add_haze import add_haze
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


def convert2float(image):
    """Transform from int image ([0,255]) to float tensor ([-1.,1.])"""
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image / 127.5) - 1.0


def build_laplacian_pyramid(image, levels=4):
    gaussian_pyramid = [image.copy()]
    for i in range(levels):
        down = cv2.pyrDown(gaussian_pyramid[-1])
        gaussian_pyramid.append(down)
    laplacian_pyramid = []
    for i in range(levels):
        h, w = gaussian_pyramid[i].shape[:2]
        up = cv2.resize(gaussian_pyramid[i + 1], (w, h), interpolation=cv2.INTER_CUBIC)
        laplacian = cv2.subtract(gaussian_pyramid[i], up, dtype=cv2.CV_16S)
        laplacian_pyramid.append(laplacian)
    return laplacian_pyramid, gaussian_pyramid[-1]


def reconstruct_from_laplacian_pyramid(base, laplacian_pyramid):
    current_img = base
    for i in range(len(laplacian_pyramid) - 1, -1, -1):
        h, w = laplacian_pyramid[i].shape[:2]
        current_img = cv2.resize(current_img, (w, h), interpolation=cv2.INTER_CUBIC)
        current_img = cv2.add(current_img.astype(np.int16), laplacian_pyramid[i], dtype=cv2.CV_16S)
    current_img = np.clip(current_img, 0, 255).astype(np.uint8)
    return current_img


def dehaze_image(input_img_bgr, model_path, image_size=256):
    """Run dehazing inference on a single BGR image array."""
    orig_h, orig_w = input_img_bgr.shape[:2]
    laplacian_pyramid, _ = build_laplacian_pyramid(input_img_bgr, levels=4)

    downscaled = cv2.resize(input_img_bgr, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    downscaled_rgb = cv2.cvtColor(downscaled, cv2.COLOR_BGR2RGB)

    with tf.Graph().as_default() as graph:
        input_image = tf.constant(downscaled_rgb, dtype=tf.uint8)
        input_image = convert2float(input_image)
        input_image = tf.reshape(input_image, [image_size, image_size, 3])

        with tf.io.gfile.GFile(model_path, 'rb') as model_file:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(model_file.read())
        [output_image] = tf.import_graph_def(graph_def,
                                              input_map={'input_image': input_image},
                                              return_elements=['output_image:0'],
                                              name='output')
        with tf.compat.v1.Session(graph=graph) as sess:
            generated = output_image.eval()

    generated_array = np.frombuffer(generated, dtype=np.uint8)
    dehazed_256 = cv2.imdecode(generated_array, cv2.IMREAD_COLOR)

    start_level_size = laplacian_pyramid[-1].shape[:2][::-1]
    upscaled = cv2.resize(dehazed_256, start_level_size, interpolation=cv2.INTER_CUBIC)
    reconstructed = reconstruct_from_laplacian_pyramid(upscaled, laplacian_pyramid)
    return reconstructed


def run_roundtrip(clear_dir, output_base, model_path, mode, intensity):
    """Run the full round-trip pipeline."""
    synth_hazy_dir = os.path.join(output_base, f"synthetic_hazy_{mode}")
    dehazed_dir = os.path.join(output_base, f"dehazed_{mode}")
    os.makedirs(synth_hazy_dir, exist_ok=True)
    os.makedirs(dehazed_dir, exist_ok=True)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    clear_images = sorted(glob.glob(os.path.join(clear_dir, '*.*')))
    clear_images = [f for f in clear_images if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp']]

    print(f"\n{'=' * 70}")
    print(f"  HAZE ROUND-TRIP PIPELINE")
    print(f"{'=' * 70}")
    print(f"  Clear Images : {clear_dir} ({len(clear_images)} images)")
    print(f"  Model        : {model_path}")
    print(f"  Haze Mode    : {mode}")
    print(f"  Intensity    : {intensity}")
    print(f"  Formula      : I(x) = J(x) * t(x) + A * (1 - t(x))")
    print(f"{'=' * 70}\n")

    results = []

    for i, clear_path in enumerate(clear_images):
        filename = os.path.basename(clear_path)
        print(f"  [{i + 1}/{len(clear_images)}] Processing {filename}...")

        # Step 1: Read clear image
        clear_img = cv2.imread(clear_path)
        if clear_img is None:
            print(f"    [ERROR] Could not read {filename}")
            continue

        # Step 2: Add synthetic haze
        hazy_img = add_haze(clear_img, mode=mode, intensity=intensity)
        hazy_path = os.path.join(synth_hazy_dir, filename)
        cv2.imwrite(hazy_path, hazy_img)

        # Step 3: Dehaze
        dehazed_img = dehaze_image(hazy_img, model_path)
        dehazed_path = os.path.join(dehazed_dir, filename)
        cv2.imwrite(dehazed_path, dehazed_img)

        # Step 4: Evaluate (compare dehazed output to original clear)
        if clear_img.shape != dehazed_img.shape:
            dehazed_img = cv2.resize(dehazed_img, (clear_img.shape[1], clear_img.shape[0]))
        clear_rgb = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        dehazed_rgb = cv2.cvtColor(dehazed_img, cv2.COLOR_BGR2RGB)

        p = psnr(clear_rgb, dehazed_rgb, data_range=255)
        s = ssim(clear_rgb, dehazed_rgb, channel_axis=2, data_range=255)
        results.append((filename, p, s))
        print(f"    PSNR={p:.4f}  SSIM={s:.4f}")

    # Print summary
    if results:
        avg_p = sum(r[1] for r in results) / len(results)
        avg_s = sum(r[2] for r in results) / len(results)
        print(f"\n{'=' * 70}")
        print(f"  ROUND-TRIP RESULTS SUMMARY ({mode} haze, intensity={intensity})")
        print(f"{'=' * 70}")
        print(f"  {'File':<30} {'PSNR':>10} {'SSIM':>10}")
        print(f"  {'-' * 30} {'-' * 10} {'-' * 10}")
        for fname, p, s in results:
            print(f"  {fname:<30} {p:>10.4f} {s:>10.4f}")
        print(f"  {'-' * 52}")
        print(f"  {'AVERAGE':<30} {avg_p:>10.4f} {avg_s:>10.4f}")
        print(f"{'=' * 70}")
        print(f"\n  Synthetic hazy images saved to : {synth_hazy_dir}")
        print(f"  Dehazed images saved to        : {dehazed_dir}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Round-trip: Clear → Add Haze → Dehaze → Evaluate")
    parser.add_argument('--clear_dir', required=True, help="Directory of clear/GT images")
    parser.add_argument('--output_dir', default='results/roundtrip', help="Base output directory")
    parser.add_argument('--model', required=True, help="Path to the .pb model file")
    parser.add_argument('--mode', default='uniform', choices=['uniform', 'depth', 'random'],
                        help="Haze synthesis mode (default: uniform)")
    parser.add_argument('--intensity', type=float, default=0.5,
                        help="Haze intensity 0.0-1.0 (default: 0.5)")
    args = parser.parse_args()

    run_roundtrip(args.clear_dir, args.output_dir, args.model, args.mode, args.intensity)
