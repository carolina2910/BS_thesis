# Neural Network Training and Testing Pipeline

A comprehensive pipeline for training and evaluating UNet and ResUNet models for image reconstruction tasks.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output Structure](#output-structure)
- [Metrics](#metrics)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements an automated pipeline for:
- Training deep learning models (UNet, ResUNet) for image reconstruction
- Testing trained models on separate test datasets
- Computing comprehensive evaluation metrics (PSNR, SSIM, RE, RMSE)
- Generating comparison visualizations and performance reports

## ✨ Features

- **Multiple Model Architectures**: UNet and ResUNet with customizable channels
- **Multiple Loss Functions**: MSE and L1 loss support
- **Automatic GPU Management**: CUDA support with memory cleanup
- **Comprehensive Metrics**: PSNR, SSIM, RE, RMSE for thorough evaluation
- **Visualization**: Automated generation of:
  - Training loss curves
  - Side-by-side image comparisons
  - Metrics comparison tables
- **Organized Output**: Structured directory hierarchy for easy result management

## 📦 Requirements

### Software Requirements
- Python 3.8+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 10GB+ free disk space

### Python Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
Pillow>=8.3.0
scikit-image>=0.18.0
```

### Custom Package
- `IPPy`: Custom utilities package for image processing and model training

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install IPPy Package
```bash
# If IPPy is in a separate repository
pip install -e path/to/IPPy

# Or if it's included
pip install -e .
```

### 5. Verify GPU Setup
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📁 Project Structure

```
project/
├── main.ipynb                          # Main pipeline notebook
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── data/
│   ├── input/
│   │   ├── Canova4/                   # Training input images
│   │   └── Canova2/                   # Training target images
│   └── target/
│       ├── Canova4_target_67932/      # Test input images
│       └── Canova2_target_67932/      # Test target images
├── results/
│   └── output_reconstruction_test_67932/
│       ├── UNet/
│       │   ├── 100epochs_6batch_MSELoss/
│       │   │   ├── model_weights.pth
│       │   │   ├── loss_curve.png
│       │   │   ├── loss_values.csv
│       │   │   ├── metrics.txt
│       │   │   ├── single_images/
│       │   │   └── comparison/
│       │   └── 100epochs_6batch_L1Loss/
│       └── ResUNet/
│           └── ...
└── IPPy/
    ├── nn/
    │   ├── models.py                  # Model architectures
    │   └── train.py                   # Training functions
    └── utilities/
        ├── data.py                    # Dataset classes
        ├── metrics.py                 # Evaluation metrics
        └── test.py                    # Testing utilities
```

## 💻 Usage

### Quick Start (Jupyter Notebook)

1. **Open the notebook**:
```bash
jupyter notebook main.ipynb
```

2. **Run all cells** or execute the main pipeline:
```python
# In the notebook
main()
```

### Step-by-Step Execution

#### 1. Configure Parameters
Edit the configuration cell in the notebook:
```python
# Data paths
INPUT_PATH_TRAIN = 'data/input/Canova4'
OUTPUT_PATH_TRAIN = 'data/input/Canova2'

# Training parameters
N_EPOCHS = 100
BATCH_SIZE = 6

# Model options
MODELS = ['UNet', 'ResUNet']
LOSS_FUNCTIONS = ['MSE', 'L1']
```

#### 2. Train Individual Model
```python
# Train specific model-loss combination
train_model('UNet', 'MSE')
```

#### 3. Test Trained Model
```python
# Test and get metrics
metrics = test_model('UNet', 'MSE')
print(f"PSNR: {metrics['psnr_pred_target']:.2f}")
print(f"SSIM: {metrics['ssim_pred_target']:.4f}")
```

#### 4. Run Complete Pipeline
```python
# Train and test all combinations
main()
```

### Command-Line Execution

Convert the notebook to a Python script:
```bash
jupyter nbconvert --to script main.ipynb
python main.py
```

## ⚙️ Configuration

### Model Architecture Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_SHAPE` | 512 | Input/output image size |
| `MIDDLE_CHANNELS` | [64, 128, 256, 512, 1024] | Channel progression |
| `FINAL_ACTIVATION` | 'sigmoid' | Output activation function |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_EPOCHS` | 100 | Number of training epochs |
| `BATCH_SIZE` | 6 | Batch size for training |
| `DEVICE` | 'cuda' (if available) | Computing device |

### Model Options

| Model | Description |
|-------|-------------|
| `UNet` | Standard U-Net architecture |
| `ResUNet` | U-Net with residual connections |

### Loss Functions

| Loss | Description | Use Case |
|------|-------------|----------|
| `MSE` | Mean Squared Error | General purpose, smooth gradients |
| `L1` | Mean Absolute Error | Robust to outliers |

## 📊 Output Structure

### For Each Model-Loss Combination

```
results/output_reconstruction_test_67932/{MODEL}/{EPOCHS}epochs_{BATCH}batch_{LOSS}Loss/
├── model_weights.pth              # Trained model weights
├── loss_curve.png                 # Training loss visualization
├── loss_values.csv                # Loss values per epoch
├── metrics.txt                    # Comprehensive metrics summary
├── single_images/                 # Individual predictions
│   ├── predicted_0.png
│   ├── predicted_1.png
│   └── ...
└── comparison/                    # Side-by-side comparisons
    ├── comparison_0.png           # [Input | Target | Prediction]
    ├── comparison_1.png
    └── ...
```

### Metrics File Format

```
Testing Results for UNet with MSE Loss
================================================================================

Number of test images: 50

Average Metrics:
--------------------------------------------------------------------------------

Input-Target Comparison:
  RE:   0.1234
  PSNR: 28.5432
  SSIM: 0.8765
  RMSE: 0.0987

Prediction-Target Comparison:
  RE:   0.0876
  PSNR: 32.1234
  SSIM: 0.9234
  RMSE: 0.0654
```

## 📈 Metrics

### Evaluation Metrics Explained

| Metric | Full Name | Range | Better | Description |
|--------|-----------|-------|--------|-------------|
| **PSNR** | Peak Signal-to-Noise Ratio | [0, ∞) dB | Higher ↑ | Measures signal quality, typically 20-50 dB |
| **SSIM** | Structural Similarity Index | [0, 1] | Higher ↑ | Perceptual similarity, considers structure |
| **RE** | Relative Error | [0, ∞) | Lower ↓ | Normalized difference, scale-independent |
| **RMSE** | Root Mean Square Error | [0, ∞) | Lower ↓ | Pixel-wise reconstruction error |

### Typical Good Values

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| PSNR | > 35 dB | 30-35 dB | 25-30 dB | < 25 dB |
| SSIM | > 0.95 | 0.90-0.95 | 0.85-0.90 | < 0.85 |
| RE | < 0.05 | 0.05-0.10 | 0.10-0.15 | > 0.15 |

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `BATCH_SIZE` (try 4 or 2)
- Reduce `DATA_SHAPE` (try 256)
- Close other GPU applications
- Use fewer middle channels

```python
# Reduced configuration
BATCH_SIZE = 2
DATA_SHAPE = 256
MIDDLE_CHANNELS = [32, 64, 128, 256, 512]
```

#### 2. Dataset Not Found
**Error**: `FileNotFoundError` or empty dataset

**Solutions**:
- Verify data paths exist
- Check file permissions
- Ensure images are in correct format (PNG, JPG)

```python
# Check data availability
import os
print(f"Training input exists: {os.path.exists(INPUT_PATH_TRAIN)}")
print(f"Files in directory: {len(os.listdir(INPUT_PATH_TRAIN))}")
```

#### 3. Import Error (IPPy)
**Error**: `ModuleNotFoundError: No module named 'IPPy'`

**Solution**:
```bash
# Install IPPy package
pip install -e path/to/IPPy

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/IPPy"
```

#### 4. Slow Training
**Problem**: Training takes too long

**Solutions**:
- Verify GPU is being used: `torch.cuda.is_available()`
- Reduce number of epochs for testing
- Use smaller dataset subset
- Enable GPU acceleration in Jupyter:
  ```
  Runtime > Change runtime type > GPU (T4/A100)
  ```

#### 5. Model Not Loading
**Error**: Model weights not found during testing

**Solution**:
```python
# Check if weights exist
import os
weights_path = 'results/.../model_weights.pth'
if not os.path.exists(weights_path):
    print("Train the model first!")
    train_model('UNet', 'MSE')
```

## 🎓 Expected Runtime

### Training Phase
- **Per model**: ~20-30 minutes (100 epochs, batch size 6, GPU)
- **All combinations**: ~2-2.5 hours (4 models)

### Testing Phase
- **Per model**: ~2-5 minutes (50 test images)
- **All combinations**: ~10-20 minutes

### Total Pipeline
- **Complete run**: ~2.5-3 hours on GPU
- **CPU only**: 10-15x longer (not recommended)

## 📝 Best Practices

### 1. Start Small
```python
# Quick test configuration
N_EPOCHS = 10
BATCH_SIZE = 2
MODELS = ['UNet']
LOSS_FUNCTIONS = ['MSE']
```

### 2. Monitor GPU Usage
```python
# Check GPU memory
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### 3. Save Checkpoints
```python
# Add to training loop
if epoch % 10 == 0:
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')
```

### 4. Visualize During Training
```python
# Plot sample predictions every N epochs
if epoch % 20 == 0:
    with torch.no_grad():
        sample_output = model(sample_input)
        visualize_prediction(sample_input, sample_output)
```

## 📧 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact: [your.email@example.com]
- Documentation: [link to docs]

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

- UNet architecture: Ronneberger et al. (2015)
- ResUNet: Zhang et al. (2018)
- IPPy utilities: [Credit contributors]

## 📚 References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
2. Zhang, Z., Liu, Q., & Wang, Y. (2018). Road Extraction by Deep Residual U-Net.

---

**Last Updated**: November 2024  
**Version**: 1.0.0