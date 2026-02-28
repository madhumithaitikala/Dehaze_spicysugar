"""
Batch evaluation script: runs PSNR/SSIM across all I-HAZE splits and prints a combined report.
"""
import cv2, os, re, sys
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def find_gt(filename, gt_dir):
    basename = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    exact = os.path.join(gt_dir, filename)
    if os.path.exists(exact): return exact
    cleaned = re.sub(r'(_hazy|_fog|_input)', '', basename)
    for e in [ext, '.png', '.jpg']:
        c = os.path.join(gt_dir, cleaned + e)
        if os.path.exists(c): return c
    return None

def eval_split(split_name, results_dir, gt_dir):
    results = []
    for f in sorted(os.listdir(results_dir)):
        if not f.lower().endswith(('.png','.jpg','.jpeg')): continue
        gt_path = find_gt(f, gt_dir)
        if gt_path is None: continue
        r = cv2.imread(os.path.join(results_dir, f))
        g = cv2.imread(gt_path)
        if r is None or g is None: continue
        if r.shape != g.shape:
            r = cv2.resize(r, (g.shape[1], g.shape[0]))
        r2 = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        g2 = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
        p = psnr(g2, r2, data_range=255)
        s = ssim(g2, r2, channel_axis=2, data_range=255)
        results.append((f, os.path.basename(gt_path), p, s))
    return results

splits = [
    ("TRAIN (25)", "results/I-HAZE/train", "data/I-HAZE/train/clear"),
    ("VAL (5)", "results/I-HAZE/val", "data/I-HAZE/val/clear"),
    ("TEST (5)", "results/I-HAZE/test", "data/I-HAZE/test/clear"),
]

all_results = []
print(f"\n{'='*80}")
print(f" I-HAZE EVALUATION REPORT - ALL SPLITS (35 Images)")
print(f"{'='*80}")

for split_name, rdir, gdir in splits:
    res = eval_split(split_name, rdir, gdir)
    all_results.extend(res)
    if not res: continue
    avg_p = sum(r[2] for r in res) / len(res)
    avg_s = sum(r[3] for r in res) / len(res)
    print(f"\n--- {split_name} ---")
    print(f" {'File':<25} {'GT':<20} {'PSNR':>8} {'SSIM':>8}")
    print(f" {'-'*25} {'-'*20} {'-'*8} {'-'*8}")
    for f, g, p, s in res:
        print(f" {f:<25} {g:<20} {p:>8.4f} {s:>8.4f}")
    print(f" {'SPLIT AVG':<46} {avg_p:>8.4f} {avg_s:>8.4f}")

if all_results:
    total_p = sum(r[2] for r in all_results) / len(all_results)
    total_s = sum(r[3] for r in all_results) / len(all_results)
    print(f"\n{'='*80}")
    print(f" OVERALL AVERAGE ({len(all_results)} images): PSNR = {total_p:.4f}  |  SSIM = {total_s:.4f}")
    print(f"{'='*80}\n")
