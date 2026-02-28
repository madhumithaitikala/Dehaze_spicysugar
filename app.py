"""
app.py - Flask Web Dashboard for Image Dehazing
Connects the existing index.html frontend to the Cycle-Dehaze and AOD-Net models.

Endpoints:
    GET  /              → Serves index.html
    POST /dehaze        → Processes hazy image and returns dehazed result
    GET  /static/<path> → Serves static files (results)

Usage:
    python app.py
    Then open http://localhost:5000 in your browser.
"""

import os
import uuid
import cv2
import numpy as np
import torch
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory, send_file
from aodnet_model import AODNet

# ==================== Configuration ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'web_results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=RESULTS_DIR)

# Suppress TF verbose logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==================== AOD-Net Loader ====================
_aodnet_model = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_aodnet():
    global _aodnet_model
    if _aodnet_model is None:
        model_path = os.path.join(MODELS_DIR, 'aodnet', 'aodnet_best.pth')
        if os.path.exists(model_path):
            _aodnet_model = AODNet().to(_device)
            _aodnet_model.load_state_dict(
                torch.load(model_path, map_location=_device, weights_only=True)
            )
            _aodnet_model.eval()
            print(f"  [AOD-Net] Loaded from {model_path}")
        else:
            print(f"  [AOD-Net] Weights not found at {model_path}")
    return _aodnet_model

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
    laplacian_pyramid, _ = build_laplacian_pyramid(img_bgr, levels=4)
    downscaled = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
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

def run_aodnet(img_bgr, image_size=256):
    model = get_aodnet()
    if model is None:
        return None
    orig_h, orig_w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(_device)
    with torch.no_grad():
        output = model(tensor)
    out_np = output.squeeze(0).cpu().numpy()
    out_np = np.clip(out_np, 0, 1)
    out_np = (out_np * 255).astype(np.uint8).transpose(1, 2, 0)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    return cv2.resize(out_bgr, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

# ==================== Ensemble ====================

def ensemble_weighted(img1, img2, w1=0.6, w2=0.4):
    return np.clip(img1.astype(np.float64) * w1 + img2.astype(np.float64) * w2, 0, 255).astype(np.uint8)

# ==================== Metrics ====================

def compute_metrics(result_bgr, gt_bgr):
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    except ImportError:
        return None

    if result_bgr.shape != gt_bgr.shape:
        result_bgr = cv2.resize(result_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]),
                                 interpolation=cv2.INTER_CUBIC)
    r = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    g = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
    p = psnr_fn(g, r, data_range=255)
    s = ssim_fn(g, r, channel_axis=2, data_range=255)
    return {'psnr': f'{p:.2f}', 'ssim': f'{s:.4f}'}

# ==================== Routes ====================

@app.route('/')
def index():
    return send_file(os.path.join(BASE_DIR, 'index.html'))

@app.route('/web_results/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route('/dehaze', methods=['POST'])
def dehaze():
    # Validate input
    if 'hazy_image' not in request.files:
        return jsonify({'error': 'No hazy image uploaded'}), 400

    hazy_file = request.files['hazy_image']
    if hazy_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    model_type = request.form.get('model_type', 'indoor')
    model_choice = request.form.get('model_choice', 'ensemble')

    # Save uploaded hazy image
    uid = str(uuid.uuid4())[:8]
    hazy_ext = os.path.splitext(hazy_file.filename)[1] or '.png'
    hazy_path = os.path.join(UPLOAD_DIR, f'{uid}_hazy{hazy_ext}')
    hazy_file.save(hazy_path)

    # Read with OpenCV
    img_bgr = cv2.imread(hazy_path)
    if img_bgr is None:
        return jsonify({'error': 'Could not read the uploaded image'}), 400

    # Select Cycle-Dehaze model based on scene type
    if model_type == 'outdoor':
        cyclegan_model = os.path.join(MODELS_DIR, 'Hazy2GT_outdoor.pb')
    else:
        cyclegan_model = os.path.join(MODELS_DIR, 'Hazy2GT_indoor.pb')

    # Route to the selected model
    model_used = model_choice
    print(f"  [Dehaze] Processing {hazy_file.filename} | Model: {model_choice} | Scene: {model_type}")

    if model_choice == 'cyclegan':
        # Cycle-Dehaze only
        final_result = run_cyclegan(img_bgr, cyclegan_model)
        model_used = f'Cycle-Dehaze ({model_type})'

    elif model_choice == 'aodnet':
        # AOD-Net only
        result_aodnet = run_aodnet(img_bgr)
        if result_aodnet is not None:
            final_result = result_aodnet
            model_used = 'AOD-Net'
        else:
            # Fallback to Cycle-Dehaze if AOD-Net not available
            final_result = run_cyclegan(img_bgr, cyclegan_model)
            model_used = f'Cycle-Dehaze ({model_type}) [AOD-Net unavailable]'

    else:
        # Ensemble (default) - combine both models
        result_cyclegan = run_cyclegan(img_bgr, cyclegan_model)
        result_aodnet = run_aodnet(img_bgr)
        if result_aodnet is not None:
            if result_cyclegan.shape != result_aodnet.shape:
                result_aodnet = cv2.resize(result_aodnet,
                                            (result_cyclegan.shape[1], result_cyclegan.shape[0]),
                                            interpolation=cv2.INTER_CUBIC)
            final_result = ensemble_weighted(result_cyclegan, result_aodnet)
            model_used = f'Ensemble: Cycle-Dehaze ({model_type}) + AOD-Net'
        else:
            final_result = result_cyclegan
            model_used = f'Cycle-Dehaze ({model_type}) [AOD-Net unavailable]'

    print(f"  [Dehaze] Using: {model_used}")

    # Save result
    result_filename = f'{uid}_dehazed.png'
    result_path = os.path.join(RESULTS_DIR, result_filename)
    cv2.imwrite(result_path, final_result)

    response = {
        'dehazed_url': f'/web_results/{result_filename}',
        'model_used': model_used
    }

    # Compute metrics if GT provided
    if 'gt_image' in request.files and request.files['gt_image'].filename != '':
        gt_file = request.files['gt_image']
        gt_path = os.path.join(UPLOAD_DIR, f'{uid}_gt{os.path.splitext(gt_file.filename)[1]}')
        gt_file.save(gt_path)
        gt_bgr = cv2.imread(gt_path)
        if gt_bgr is not None:
            scores = compute_metrics(final_result, gt_bgr)
            if scores:
                response['scores'] = scores
                print(f"  [Metrics] PSNR={scores['psnr']}, SSIM={scores['ssim']}")

    print(f"  [Dehaze] Result saved: {result_path}")
    return jsonify(response)

# ==================== Main ====================

if __name__ == '__main__':
    print(f"\n{'=' * 50}")
    print(f"  AI Image Dehazing Dashboard")
    print(f"{'=' * 50}")
    print(f"  Models: Cycle-Dehaze + AOD-Net (Ensemble)")
    print(f"  Device: {_device}")
    print(f"  URL: http://localhost:5000")
    print(f"{'=' * 50}\n")

    # Pre-load AOD-Net
    get_aodnet()

    app.run(host='0.0.0.0', port=5000, debug=False)
