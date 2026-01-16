#!/bin/bash
echo "=== DragGAN Colab Setup Started ==="

# 1. Install system dependencies (needed for OpenCV)
echo "Installing system dependencies..."
apt-get update && apt-get install -y libgl1-mesa-glx

# 2. Install Python dependencies
echo "Installing python dependencies (this may take a minute)..."
pip install -r requirements.txt

# 3. Download pretrained models
echo "Downloading pretrained models (approx 2GB)..."
python scripts/download_model.py

# 4. Verify installation
echo "Verifying environment..."
python check_env.py

echo "=== Setup Complete! ==="
echo "You can now run: python visualizer_drag_gradio.py --share"
