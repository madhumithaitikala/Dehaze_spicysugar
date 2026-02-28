"""
AOD-Net Inference Script
Dehazes images using a trained AOD-Net model.

Usage:
    python aodnet_inference.py --input_dir data/I-HAZE/test/hazy --output_dir results/aodnet/I-HAZE/test --model models/aodnet/aodnet_best.pth
"""

import argparse
import os
import glob
import cv2
import numpy as np
import torch
from aodnet_model import AODNet


def dehaze_image(model, img_bgr, device):
    """Dehaze a single BGR image using AOD-Net."""
    orig_h, orig_w = img_bgr.shape[:2]

    # Preprocess: BGR -> RGB, normalize to [0, 1], HWC -> CHW
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(tensor)

    # Postprocess: CHW -> HWC, [0,1] -> [0,255], RGB -> BGR
    output_np = output.squeeze(0).cpu().numpy()
    output_np = np.clip(output_np, 0, 1)
    output_np = (output_np * 255).astype(np.uint8)
    output_np = output_np.transpose(1, 2, 0)  # CHW -> HWC
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

    # Resize back to original dimensions if needed
    if output_bgr.shape[:2] != (orig_h, orig_w):
        output_bgr = cv2.resize(output_bgr, (orig_w, orig_h))

    return output_bgr


def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  AOD-Net Inference")
    print(f"  Device: {device}")
    print(f"  Model: {args.model}\n")

    # Load model
    model = AODNet().to(device)
    state_dict = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Process images
    os.makedirs(args.output_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(args.input_dir, '*.*')))

    count = 0
    for img_path in images:
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [ERROR] Could not read {img_path}")
            continue

        # For AOD-Net, we resize to a manageable size for inference, then resize back
        orig_h, orig_w = img.shape[:2]
        # Resize to inference size
        img_resized = cv2.resize(img, (args.image_size, args.image_size))

        dehazed = dehaze_image(model, img_resized, device)

        # Resize back to original
        dehazed = cv2.resize(dehazed, (orig_w, orig_h))

        out_path = os.path.join(args.output_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, dehazed)
        print(f"  {os.path.basename(img_path)} -> {out_path}")
        count += 1

    print(f"\n  Processed {count} images.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dehaze images using AOD-Net")
    parser.add_argument('--input_dir', required=True, help="Input hazy images directory")
    parser.add_argument('--output_dir', required=True, help="Output dehazed images directory")
    parser.add_argument('--model', required=True, help="Path to trained AOD-Net .pth weights")
    parser.add_argument('--image_size', type=int, default=256, help="Inference image size (default: 256)")
    args = parser.parse_args()

    inference(args)
