"""
ensemble_dehaze.py - Ensemble dehazing by combining Cycle-Dehaze and AOD-Net outputs.

Fusion strategies:
    1. weighted_avg  : Weighted pixel-level average (default: 0.6 Cycle-Dehaze + 0.4 AOD-Net)
    2. max_clarity   : Per-pixel selection based on local contrast (pick the clearer patch)
    3. laplacian_blend: Multi-scale Laplacian blending (low freq from one, high freq from another)

Usage:
    python ensemble_dehaze.py --input_dir data/I-HAZE/test/hazy --output_dir results/ensemble/I-HAZE/test --cyclegan_model models/Hazy2GT_indoor.pb --aodnet_model models/aodnet/aodnet_best.pth --strategy weighted_avg
"""

import argparse
import os
import glob
import cv2
import numpy as np
import torch
import tensorflow as tf
from aodnet_model import AODNet


# ==================== Cycle-Dehaze Inference ====================

def convert2float_tf(image):
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


def run_cyclegan(img_bgr, model_path, image_size=256):
    """Run Cycle-Dehaze on a single BGR image."""
    laplacian_pyramid, _ = build_laplacian_pyramid(img_bgr, levels=4)
    downscaled = cv2.resize(img_bgr, (image_size, image_size))
    downscaled_rgb = cv2.cvtColor(downscaled, cv2.COLOR_BGR2RGB)

    with tf.Graph().as_default() as graph:
        input_image = tf.constant(downscaled_rgb, dtype=tf.uint8)
        input_image = convert2float_tf(input_image)
        input_image = tf.reshape(input_image, [image_size, image_size, 3])
        with tf.io.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        [output_image] = tf.import_graph_def(graph_def,
                                              input_map={'input_image': input_image},
                                              return_elements=['output_image:0'],
                                              name='output')
        with tf.compat.v1.Session(graph=graph) as sess:
            generated = output_image.eval()

    arr = np.frombuffer(generated, dtype=np.uint8)
    dehazed_256 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    start_size = laplacian_pyramid[-1].shape[:2][::-1]
    upscaled = cv2.resize(dehazed_256, start_size, interpolation=cv2.INTER_CUBIC)
    return reconstruct_from_laplacian_pyramid(upscaled, laplacian_pyramid)


# ==================== AOD-Net Inference ====================

def run_aodnet(img_bgr, model, device, image_size=256):
    """Run AOD-Net on a single BGR image."""
    orig_h, orig_w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (image_size, image_size))
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    out_np = output.squeeze(0).cpu().numpy()
    out_np = np.clip(out_np, 0, 1)
    out_np = (out_np * 255).astype(np.uint8).transpose(1, 2, 0)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    return cv2.resize(out_bgr, (orig_w, orig_h))


# ==================== Fusion Strategies ====================

def weighted_average(img1, img2, w1=0.6, w2=0.4):
    """Simple weighted pixel average."""
    return np.clip(img1.astype(np.float64) * w1 + img2.astype(np.float64) * w2, 0, 255).astype(np.uint8)


def max_clarity_fusion(img1, img2, block_size=16):
    """Pick patches from the image with higher local contrast (Laplacian variance)."""
    h, w = img1.shape[:2]
    result = np.zeros_like(img1)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y2 = min(y + block_size, h)
            x2 = min(x + block_size, w)

            patch1 = img1[y:y2, x:x2]
            patch2 = img2[y:y2, x:x2]

            # Compute Laplacian variance (measure of sharpness/clarity)
            lap1 = cv2.Laplacian(cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            lap2 = cv2.Laplacian(cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

            if lap1 >= lap2:
                result[y:y2, x:x2] = patch1
            else:
                result[y:y2, x:x2] = patch2

    return result


def laplacian_blend(img1, img2, levels=5):
    """
    Multi-scale Laplacian blending:
    Takes low-frequency content from img1 (Cycle-Dehaze, better color)
    and high-frequency details from img2 (AOD-Net, preserves structure).
    """
    # Build Gaussian pyramids
    g1 = [img1.astype(np.float64)]
    g2 = [img2.astype(np.float64)]
    for _ in range(levels):
        g1.append(cv2.pyrDown(g1[-1]))
        g2.append(cv2.pyrDown(g2[-1]))

    # Build Laplacian pyramids
    l1, l2 = [], []
    for i in range(levels):
        h, w = g1[i].shape[:2]
        l1.append(g1[i] - cv2.resize(g1[i + 1], (w, h)))
        l2.append(g2[i] - cv2.resize(g2[i + 1], (w, h)))

    # Blend: low freq from img1, high freq from weighted mix
    blended = []
    for i in range(levels):
        if i < 2:
            # High frequency: blend both
            blended.append(l1[i] * 0.5 + l2[i] * 0.5)
        else:
            # Low frequency: favor Cycle-Dehaze
            blended.append(l1[i] * 0.7 + l2[i] * 0.3)

    # Reconstruct
    result = g1[-1] * 0.6 + g2[-1] * 0.4  # base from weighted avg
    for i in range(levels - 1, -1, -1):
        h, w = blended[i].shape[:2]
        result = cv2.resize(result, (w, h)) + blended[i]

    return np.clip(result, 0, 255).astype(np.uint8)


# ==================== Main ====================

def ensemble(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    print(f"\n{'=' * 60}")
    print(f"  ENSEMBLE DEHAZING PIPELINE")
    print(f"{'=' * 60}")
    print(f"  Model 1 : Cycle-Dehaze ({args.cyclegan_model})")
    print(f"  Model 2 : AOD-Net ({args.aodnet_model})")
    print(f"  Strategy: {args.strategy}")
    print(f"  Device  : {device}")
    print(f"{'=' * 60}\n")

    # Load AOD-Net
    aodnet = AODNet().to(device)
    aodnet.load_state_dict(torch.load(args.aodnet_model, map_location=device, weights_only=True))
    aodnet.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(args.input_dir, '*.*')))
    images = [f for f in images if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp']]

    for i, img_path in enumerate(images):
        filename = os.path.basename(img_path)
        print(f"  [{i + 1}/{len(images)}] {filename}...")

        img = cv2.imread(img_path)
        if img is None:
            print(f"    [ERROR] Could not read.")
            continue

        # Run both models
        result_cyclegan = run_cyclegan(img, args.cyclegan_model)
        result_aodnet = run_aodnet(img, aodnet, device)

        # Ensure same size
        if result_cyclegan.shape != result_aodnet.shape:
            result_aodnet = cv2.resize(result_aodnet,
                                        (result_cyclegan.shape[1], result_cyclegan.shape[0]))

        # Fuse
        if args.strategy == 'weighted_avg':
            fused = weighted_average(result_cyclegan, result_aodnet, w1=0.6, w2=0.4)
        elif args.strategy == 'max_clarity':
            fused = max_clarity_fusion(result_cyclegan, result_aodnet)
        elif args.strategy == 'laplacian_blend':
            fused = laplacian_blend(result_cyclegan, result_aodnet)
        else:
            fused = weighted_average(result_cyclegan, result_aodnet)

        out_path = os.path.join(args.output_dir, filename)
        cv2.imwrite(out_path, fused)
        print(f"    Saved: {out_path}")

    print(f"\n  Ensemble complete! {len(images)} images processed.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ensemble Dehazing: Cycle-Dehaze + AOD-Net")
    parser.add_argument('--input_dir', required=True, help="Input hazy images directory")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--cyclegan_model', default='models/Hazy2GT_indoor.pb',
                        help="Cycle-Dehaze model path")
    parser.add_argument('--aodnet_model', default='models/aodnet/aodnet_best.pth',
                        help="AOD-Net model path")
    parser.add_argument('--strategy', default='weighted_avg',
                        choices=['weighted_avg', 'max_clarity', 'laplacian_blend'],
                        help="Fusion strategy (default: weighted_avg)")
    args = parser.parse_args()

    ensemble(args)
