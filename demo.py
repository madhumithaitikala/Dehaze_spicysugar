import tensorflow as tf
import cv2
import numpy as np
import argparse
import os
import glob

def convert2float(image):
    """ Transform from int image ([0,255]) to float tensor ([-1.,1.]) """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image / 127.5) - 1.0

def build_laplacian_pyramid(image, levels=4):
    """
    Builds a Laplacian pyramid using OpenCV. 
    Equivalent to MATLAB's impyramid(A, 'reduce').
    """
    gaussian_pyramid = [image.copy()]
    for i in range(levels):
        # Scale down
        down = cv2.pyrDown(gaussian_pyramid[-1])
        gaussian_pyramid.append(down)
        
    laplacian_pyramid = []
    # Calculate difference between gaussian levels and upscaled next levels
    for i in range(levels):
        h, w = gaussian_pyramid[i].shape[:2]
        # upscale the next (smaller) gaussian level to match this one
        up = cv2.resize(gaussian_pyramid[i+1], (w, h), interpolation=cv2.INTER_CUBIC)
        # Compute laplacian detail map (high freq)
        laplacian = cv2.subtract(gaussian_pyramid[i], up, dtype=cv2.CV_16S)
        laplacian_pyramid.append(laplacian)
        
    return laplacian_pyramid, gaussian_pyramid[-1]

def reconstruct_from_laplacian_pyramid(base, laplacian_pyramid):
    """
    Reconstructs the image from a base and Laplacian details.
    """
    current_img = base
    # Reconstruct from smallest to largest
    for i in range(len(laplacian_pyramid) - 1, -1, -1):
        h, w = laplacian_pyramid[i].shape[:2]
        # Upscale
        current_img = cv2.resize(current_img, (w, h), interpolation=cv2.INTER_CUBIC)
        # Add high frequency details
        current_img = cv2.add(current_img.astype(np.int16), laplacian_pyramid[i], dtype=cv2.CV_16S)
        
    # Clip back to valid range
    current_img = np.clip(current_img, 0, 255).astype(np.uint8)
    return current_img

def inference(input_image_path, output_image_path, model_path, image_size=256):
    print(f"Processing: {input_image_path} -> {output_image_path}")
    
    # 1. Start Pre-processing (equivalent to resize_im.m)
    orig_img = cv2.imread(input_image_path)
    if orig_img is None:
        print(f"Error loading {input_image_path}. Skipping.")
        return
    
    # Needs to match what impyramid and resize_im was given
    orig_h, orig_w = orig_img.shape[:2]
    
    # Compute laplacian pyramid from original high-res image
    laplacian_pyramid, smallest_gaussian = build_laplacian_pyramid(orig_img, levels=4)
        
    # Resize down to 256x256 for the neural network
    downscaled_img = cv2.resize(orig_img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    # Convert BGR (OpenCV) to RGB (TF expects RGB usually if trained normally)
    downscaled_img_rgb = cv2.cvtColor(downscaled_img, cv2.COLOR_BGR2RGB)
    
    # TF preprocessing
    with tf.Graph().as_default() as graph:
        # Recreate inference.py tf input conversions
        input_image = tf.constant(downscaled_img_rgb, dtype=tf.uint8)
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
            
    # Inference returns a JPEG stream string (because in the original code output_image is assumed to be encoded jpeg bytes)
    # Actually wait, let's look at inference.py...
    # `with open(FLAGS.output, 'wb') as f: f.write(generated)` means output_image.eval() returns bytes of an encoded JPEG!
    generated_array = np.frombuffer(generated, dtype=np.uint8)
    dehazed_256 = cv2.imdecode(generated_array, cv2.IMREAD_COLOR)
    dehazed_256 = cv2.cvtColor(dehazed_256, cv2.COLOR_BGR2RGB) # Decode is BGR, graph returns standard color? 
    # Usually the model output was saved as JPG from tf.image.encode_jpeg, we'll keep it in BGR for OpenCV
    
    # 3. Upscaling (equivalent to laplacian.m)
    # "Cu4 = imresize(Cu3, [size(A,1), size(A,2)]) + L1;"
    # Actually, the original implementation resizes the C (dehazed 256x256) up through the laplacian pyramid sizes.
    # We will reconstruct up through the stored Laplacian pyramid sizes instead of building a separate downscaled one for C.
    
    # Initial step: Match dehazed size to deepest pyramid level (smallest)
    start_level_size = laplacian_pyramid[-1].shape[:2][::-1] # Width, Height
    upscaled = cv2.resize(dehazed_256, start_level_size, interpolation=cv2.INTER_CUBIC)
    
    # Now reconstruct by adding the Laplacian levels from the original hazy image
    reconstructed_img = reconstruct_from_laplacian_pyramid(upscaled, laplacian_pyramid)

    # Save final result
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, reconstructed_img)
    print(f"Successfully saved {output_image_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dehaze images safely in pure Python.")
    parser.add_argument('--input_dir', required=True, help="Path to input images directory.")
    parser.add_argument('--output_dir', required=True, help="Path to save output images.")
    parser.add_argument('--model', required=True, help="Path to the trained .pb model.")
    args = parser.parse_args()

    # Find images in input_dir
    images = glob.glob(os.path.join(args.input_dir, '*.*'))
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Turn off TF V2 debug printing to avoid flooding
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    for img_path in images:
        ext = os.path.splitext(img_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            out_path = os.path.join(args.output_dir, os.path.basename(img_path))
            inference(img_path, out_path, args.model)
