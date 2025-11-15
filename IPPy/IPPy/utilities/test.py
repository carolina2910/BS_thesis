import os
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List
from .metrics import *
from ._utilities import *
import pandas as pd

def save_image(image_tensor: torch.Tensor, path: str) -> None:
    """
    Save a tensor as an image.
    
    Args:
        image_tensor: Tensor containing image data
        path: Path to save the image
    """
    image_array = (image_tensor.squeeze().numpy() * 255).astype('uint8')
    image_pil = Image.fromarray(image_array)
    image_pil.save(path)


def save_comparison_plot(
    input_img: torch.Tensor,
    target_img: torch.Tensor,
    predicted_img: torch.Tensor,
    save_path: str,
    index: int
) -> None:
    """
    Create and save a comparison plot of input, target, and predicted images.
    
    Args:
        input_img: Input image tensor
        target_img: Target image tensor
        predicted_img: Predicted image tensor
        save_path: Path to save the plot
        index: Image index for title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(input_img.squeeze(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(target_img.squeeze(), cmap='gray')
    axes[1].set_title('Target Image')
    axes[1].axis('off')
    
    axes[2].imshow(predicted_img.squeeze(), cmap='gray')
    axes[2].set_title('Predicted Image')
    axes[2].axis('off')
    
    plt.suptitle(f'Image Comparison - Index {index}')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def calculate_metrics(
    input_img: torch.Tensor,
    target_img: torch.Tensor,
    predicted_img: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate all metrics for image comparison.
    
    Args:
        input_img: Input image tensor
        target_img: Target image tensor
        predicted_img: Predicted image tensor
    
    Returns:
        Dictionary containing all metric values
    """
    return {
        're_input_target': RE(input_img, target_img),
        'psnr_input_target': PSNR(input_img, target_img),
        'ssim_input_target': SSIM(input_img, target_img),
        'rmse_input_target': RMSE(input_img, target_img),
        're_pred_target': RE(predicted_img, target_img),
        'psnr_pred_target': PSNR(predicted_img, target_img),
        'ssim_pred_target': SSIM(predicted_img, target_img),
        'rmse_pred_target': RMSE(predicted_img, target_img)
    }





# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_loss_comparison(loss_functions: list[str], models: list[str], get_output_dir) -> None:
    """
    Create comparison plots of loss curves for all model/loss combinations.
    
    Args:
        loss_functions: List of loss function names
        models: List of model names
        get_output_dir: Function to get output directory for a given model and loss function
    """
    print("\n" + "="*80)
    print("CREATING LOSS COMPARISON PLOTS")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for idx, loss_fn_name in enumerate(loss_functions):
        ax = axes[idx]
        
        for model_name in models:
            results_dir = get_output_dir(model_name, loss_fn_name)
            loss_file = os.path.join(results_dir, 'loss_values.csv')
            
            if os.path.exists(loss_file):
                loss_df = pd.read_csv(loss_file)
                ax.plot(loss_df, label=model_name, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title(f'{loss_fn_name} Loss Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_path = 'results/loss_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Loss comparison plot saved to: {comparison_path}")


def create_metrics_comparison_table(loss_functions: list[str], models: list[str], get_output_dir) -> pd.DataFrame:
    """
    Create a comparison table of metrics for all models.
    
    Args:
        loss_functions: List of loss function names
        models: List of model names
        get_output_dir: Function to get output directory for a given model and loss function
    
    Returns:
        DataFrame containing metrics comparison
    """
    results = []
    
    for model_name in models:
        for loss_fn_name in loss_functions:
            results_dir = get_output_dir(model_name, loss_fn_name)
            metrics_file = os.path.join(results_dir, 'metrics.txt')
            
            if os.path.exists(metrics_file):
                # Parse metrics file
                with open(metrics_file, 'r') as f:
                    content = f.read()
                    
                # Extract metrics (simplified parsing)
                result_dict = {
                    'Model': model_name,
                    'Loss Function': loss_fn_name
                }
                results.append(result_dict)
    
    df = pd.DataFrame(results)
    print("\nMetrics Comparison Table:")
    print(df.to_string(index=False))
    
    return df