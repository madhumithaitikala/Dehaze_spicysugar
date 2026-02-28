import argparse
import os
import glob
import re
import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
except ImportError:
    print("scikit-image is required. Installing now...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr


def find_gt_match(result_filename, gt_dir):
    """
    Finds the matching ground truth file for a given result filename.
    Handles naming conventions like:
      - ih31_hazy.png (result) -> ih31.png (GT)
      - 31.png (result) -> 31.png (GT)
      - same name match
    """
    basename = os.path.splitext(result_filename)[0]
    ext = os.path.splitext(result_filename)[1]

    # Strategy 1: Exact filename match
    exact_path = os.path.join(gt_dir, result_filename)
    if os.path.exists(exact_path):
        return exact_path

    # Strategy 2: Strip common suffixes like '_hazy', '_fog', '_input'
    cleaned = re.sub(r'(_hazy|_fog|_input|_hazey|_smog)', '', basename)
    for candidate_ext in [ext, '.png', '.jpg', '.jpeg', '.bmp']:
        candidate = os.path.join(gt_dir, cleaned + candidate_ext)
        if os.path.exists(candidate):
            return candidate

    # Strategy 3: Try matching by leading digits only
    digits = re.search(r'(\d+)', basename)
    if digits:
        digit_str = digits.group(1)
        for gt_file in os.listdir(gt_dir):
            gt_digits = re.search(r'(\d+)', gt_file)
            if gt_digits and gt_digits.group(1) == digit_str:
                return os.path.join(gt_dir, gt_file)

    return None


def evaluate(results_dir, gt_dir):
    print(f"\n{'='*50}")
    print(f"  PSNR & SSIM Evaluation")
    print(f"{'='*50}")
    print(f"  Results : {results_dir}")
    print(f"  GT      : {gt_dir}")
    print(f"{'='*50}\n")

    result_images = sorted(glob.glob(os.path.join(results_dir, '*.*')))

    total_ssim = 0.0
    total_psnr = 0.0
    valid_count = 0
    results_log = []

    for res_path in result_images:
        filename = os.path.basename(res_path)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            continue

        gt_path = find_gt_match(filename, gt_dir)

        if gt_path is None:
            print(f"  [SKIP] {filename} — no GT match found")
            continue

        res_img = cv2.imread(res_path)
        gt_img = cv2.imread(gt_path)

        if res_img is None or gt_img is None:
            print(f"  [ERROR] Could not read {filename} or its GT. Skipping.")
            continue

        # Resize result to match GT dimensions if needed
        if res_img.shape != gt_img.shape:
            res_img = cv2.resize(res_img, (gt_img.shape[1], gt_img.shape[0]),
                                 interpolation=cv2.INTER_CUBIC)

        # Convert BGR to RGB for metrics
        res_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        current_ssim = ssim(gt_rgb, res_rgb, channel_axis=2, data_range=255)
        current_psnr = psnr(gt_rgb, res_rgb, data_range=255)

        results_log.append({
            'file': filename,
            'gt': os.path.basename(gt_path),
            'psnr': current_psnr,
            'ssim': current_ssim
        })

        total_ssim += current_ssim
        total_psnr += current_psnr
        valid_count += 1

    # Print results table
    if valid_count > 0:
        print(f"  {'Result File':<25} {'GT File':<20} {'PSNR':>10} {'SSIM':>10}")
        print(f"  {'-'*25} {'-'*20} {'-'*10} {'-'*10}")
        for r in results_log:
            print(f"  {r['file']:<25} {r['gt']:<20} {r['psnr']:>10.4f} {r['ssim']:>10.4f}")

        avg_psnr = total_psnr / valid_count
        avg_ssim = total_ssim / valid_count
        print(f"\n  {'='*67}")
        print(f"  AVERAGE ({valid_count} images):{'':>18} {avg_psnr:>10.4f} {avg_ssim:>10.4f}")
        print(f"  {'='*67}\n")
    else:
        print("\n  No matching image pairs found to evaluate.")
        print("  Make sure the result filenames can be matched to GT filenames.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Dehazing with SSIM and PSNR.")
    parser.add_argument('--results_dir', required=True, help="Path to dehazed output images.")
    parser.add_argument('--gt_dir', required=True, help="Path to Ground Truth (clear) images.")
    args = parser.parse_args()

    evaluate(args.results_dir, args.gt_dir)
