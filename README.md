# Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing

## Prerequisites

* TensorFlow 1.4.1 or later
* Python 3
* MATLAB 

Our code is tested under Ubuntu 16.04 environment with Titan X GPUs.

## Demo

* Test the model for Track 1: Indoor

```sh
 sh demo.sh data/indoor results/indoor models/Hazy2GT_indoor.pb
```

* Test the model for Track 2: Outdoor

```sh
sh demo.sh data/outdoor results/outdoor models/Hazy2GT_outdoor.pb
```

*  You can use this model for your own images. 

```sh
sh demo.sh input_folder output_folder model_name
```

## Web Dashboard

We have provided a Flask-based web dashboard to interactively test the dehazing models (Cycle-Dehaze and AOD-Net).

To run the dashboard:

1. Install the required Python packages:
   ```sh
   pip install flask opencv-python torch torchvision tensorflow numpy skimage
   ```

2. Run the Flask application:
   ```sh
   python app.py
   ```

3. Open your web browser and navigate to `http://localhost:5000` to interact with the dashboard.


