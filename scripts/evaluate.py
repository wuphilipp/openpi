#!/usr/bin/env python

"""
Evaluate a trained policy on validation data.
"""

import dataclasses
import functools
import logging
import platform
from typing import Any
import os

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tqdm
import pandas as pd
from pathlib import Path

import openpi.models.model as _model
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def load_trained_policy(
    config: _config.TrainConfig, 
    checkpoint_dir: str,
    checkpoint_step: int | None = None
) -> _policy.Policy:
    """Load a trained policy using the inference-friendly policy loader."""
    
    # If checkpoint_step is specified, append it to the directory
    if checkpoint_step is not None:
        checkpoint_path = f"{checkpoint_dir}/{checkpoint_step}"
    else:
        checkpoint_path = checkpoint_dir
    
    logging.info(f"Loading from checkpoint path: {checkpoint_path}")
    
    # Use the policy config loader which handles device/sharding differences better
    policy = _policy_config.create_trained_policy(
        config,
        checkpoint_path,
        default_prompt=None
    )
    
    return policy


def evaluate_policy(
    policy: _policy.Policy,
    observation: _model.Observation,
    actions_gt: _model.Actions,
    rng: at.KeyArrayLike,
) -> tuple[_model.Actions, dict[str, float]]:
    """Evaluate policy on a single batch and compute metrics."""
    
    # Convert observation to the format expected by policy
    # The observation is already structured properly from the data loader
    
    # Get the first image key to determine batch size
    first_image_key = next(iter(observation.images.keys()))
    batch_size = observation.images[first_image_key].shape[0]
    
    # Get policy predictions for the batch
    actions_pred_list = []
    for i in range(batch_size):
        # Extract single observation from batch
        single_obs_images = {key: jnp.array(value[i]) for key, value in observation.images.items()}
        single_obs_image_masks = {key: jnp.array(value[i]) for key, value in observation.image_masks.items()}
        single_obs_state = jnp.array(observation.state[i])
        
        # Create single observation
        single_obs = _model.Observation(
            images=single_obs_images,
            image_masks=single_obs_image_masks,
            state=single_obs_state,
            tokenized_prompt=jnp.array(observation.tokenized_prompt[i]) if observation.tokenized_prompt is not None else None,
            tokenized_prompt_mask=jnp.array(observation.tokenized_prompt_mask[i]) if observation.tokenized_prompt_mask is not None else None,
            token_ar_mask=jnp.array(observation.token_ar_mask[i]) if observation.token_ar_mask is not None else None,
            token_loss_mask=jnp.array(observation.token_loss_mask[i]) if observation.token_loss_mask is not None else None,
        )
        
        # Get policy prediction using infer method
        # Convert structured observation back to dictionary format that policy expects
        single_obs_dict = single_obs.to_dict()
        
        result = policy.infer(single_obs_dict)
        action_pred = result['actions']  # Extract actions from result
        actions_pred_list.append(action_pred)
    
    # Stack predictions back into batch format
    actions_pred = jnp.stack(actions_pred_list, axis=0)
    
    # Compute metrics
    actions_gt_np = np.array(actions_gt)
    actions_pred_np = np.array(actions_pred)
    
    # Handle shape mismatch if needed
    if actions_gt_np.shape != actions_pred_np.shape:
        logging.warning(f"Shape mismatch detected. GT: {actions_gt_np.shape}, Pred: {actions_pred_np.shape}")
        # If predicted actions have fewer dimensions, we might need to handle this
        min_samples = min(actions_gt_np.size, actions_pred_np.size)
        gt_flat = actions_gt_np.flatten()[:min_samples]
        pred_flat = actions_pred_np.flatten()[:min_samples]
    else:
        # Flatten for metric computation
        gt_flat = actions_gt_np.reshape(-1)
        pred_flat = actions_pred_np.reshape(-1)
    
    metrics = {
        "mse": float(mean_squared_error(gt_flat, pred_flat)),
        "mae": float(mean_absolute_error(gt_flat, pred_flat)),
        "rmse": float(np.sqrt(mean_squared_error(gt_flat, pred_flat))),
        "r2": float(1 - np.sum((gt_flat - pred_flat) ** 2) / np.sum((gt_flat - np.mean(gt_flat)) ** 2)),
    }
    
    return actions_pred, metrics


def plot_action_comparison(
    actions_gt: np.ndarray,
    actions_pred: np.ndarray,
    save_path: str,
    action_names: list[str] | None = None,
    sample_indices: list[int] | None = None,
):
    """Create comprehensive plots comparing ground truth vs predicted actions."""
    
    batch_size, action_horizon, gt_action_dim = actions_gt.shape
    pred_action_dim = actions_pred.shape[2]
    
    # Use the smaller action dimension for plotting
    plot_action_dim = min(gt_action_dim, pred_action_dim)
    
    if action_names is None:
        action_names = [f"Action_{i}" for i in range(plot_action_dim)]
    else:
        action_names = action_names[:plot_action_dim]  # Truncate to available dimensions
    
    if sample_indices is None:
        # Select a few random samples to plot
        sample_indices = np.random.choice(batch_size, min(4, batch_size), replace=False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(sample_indices), plot_action_dim, figsize=(4*plot_action_dim, 3*len(sample_indices)))
    if len(sample_indices) == 1:
        axes = axes.reshape(1, -1)
    if plot_action_dim == 1:
        axes = axes.reshape(-1, 1)
    
    for i, sample_idx in enumerate(sample_indices):
        for j in range(plot_action_dim):
            ax = axes[i, j]
            
            # Plot ground truth and predicted actions
            time_steps = np.arange(action_horizon)
            ax.plot(time_steps, actions_gt[sample_idx, :, j], 'b-', label='Ground Truth', linewidth=2)
            ax.plot(time_steps, actions_pred[sample_idx, :, j], 'r--', label='Predicted', linewidth=2)
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Action Value')
            ax.set_title(f'Sample {sample_idx} - {action_names[j]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_heatmap(
    actions_gt: np.ndarray,
    actions_pred: np.ndarray,
    save_path: str,
    action_names: list[str] | None = None,
):
    """Create heatmap of prediction errors across time and action dimensions."""
    
    batch_size, action_horizon, gt_action_dim = actions_gt.shape
    pred_action_dim = actions_pred.shape[2]
    
    # Use the smaller action dimension for error computation
    plot_action_dim = min(gt_action_dim, pred_action_dim)
    
    if action_names is None:
        action_names = [f"Action_{i}" for i in range(plot_action_dim)]
    else:
        action_names = action_names[:plot_action_dim]
    
    # Truncate both arrays to the same dimensions for error computation
    actions_gt_truncated = actions_gt[:, :, :plot_action_dim]
    actions_pred_truncated = actions_pred[:, :, :plot_action_dim]
    
    # Compute absolute errors
    errors = np.abs(actions_gt_truncated - actions_pred_truncated)
    
    # Average errors across batch
    avg_errors = np.mean(errors, axis=0)  # Shape: (action_horizon, plot_action_dim)
    
    # Create heatmap
    plt.figure(figsize=(max(8, plot_action_dim), max(6, action_horizon // 2)))
    sns.heatmap(
        avg_errors.T,  # Transpose so actions are on y-axis
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=[f'T{i}' for i in range(action_horizon)],
        yticklabels=action_names,
        cbar_kws={'label': 'Mean Absolute Error'}
    )
    plt.title('Prediction Error Heatmap\n(Averaged across batch)')
    plt.xlabel('Time Step')
    plt.ylabel('Action Dimension')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_distribution(
    actions_gt: np.ndarray,
    actions_pred: np.ndarray,
    save_path: str,
):
    """Plot distribution of prediction errors."""
    
    # Handle dimension mismatch - use only the overlapping dimensions
    min_action_dim = min(actions_gt.shape[-1], actions_pred.shape[-1])
    actions_gt_trimmed = actions_gt[:, :, :min_action_dim]
    actions_pred_trimmed = actions_pred[:, :, :min_action_dim]
    
    # Compute errors
    errors = (actions_pred_trimmed - actions_gt_trimmed).flatten()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of errors
    ax1.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Prediction Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Distribution of Prediction Errors\n(Using {min_action_dim}/{actions_gt.shape[-1]} action dims)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot of absolute errors by action dimension
    abs_errors_by_dim = []
    for i in range(min_action_dim):
        abs_errors_by_dim.append(np.abs(actions_pred_trimmed[:, :, i] - actions_gt_trimmed[:, :, i]).flatten())
    
    ax2.boxplot(abs_errors_by_dim, positions=range(min_action_dim))
    ax2.set_xlabel('Action Dimension')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title(f'Error Distribution by Action Dimension\n({min_action_dim} dims)')
    ax2.set_xticks(range(min_action_dim))
    ax2.set_xticklabels([f'A{i}' for i in range(min_action_dim)])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_metrics_plot(
    all_metrics: list[dict[str, float]],
    save_path: str,
):
    """Create a summary plot of all evaluation metrics."""
    
    # Aggregate metrics
    metric_names = list(all_metrics[0].keys())
    aggregated = {}
    
    for metric in metric_names:
        values = [batch_metrics[metric] for batch_metrics in all_metrics]
        aggregated[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(metric_names))
    means = [aggregated[metric]['mean'] for metric in metric_names]
    stds = [aggregated[metric]['std'] for metric in metric_names]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, color='skyblue', edgecolor='black')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.4f}±{std:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Model Evaluation Metrics Summary')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return aggregated


# Apply video loading patch immediately
def patch_video_loading():
    """Patch LeRobot to use pyav instead of torchcodec for video loading."""
    try:
        from lerobot.common.datasets import video_utils
        
        # Patch the default codec function to return pyav instead of torchcodec
        def patched_get_safe_default_codec():
            """Always return pyav as the default codec."""
            return "pyav"
        
        # Replace the function in the module
        video_utils.get_safe_default_codec = patched_get_safe_default_codec
        
        # Also patch the decode_video_frames function directly
        original_decode = video_utils.decode_video_frames
        def patched_decode_video_frames(video_path, timestamps, tolerance_s, backend=None):
            """Force pyav backend."""
            return video_utils.decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend="pyav")
        
        video_utils.decode_video_frames = patched_decode_video_frames
        
        logging.info("Successfully patched video loading to use pyav backend")
        return True
    except Exception as e:
        logging.warning(f"Failed to patch video loading: {e}")
        return False

# Apply patch immediately on import
patch_video_loading()


def main(config: _config.TrainConfig, num_eval_batches: int = 10, checkpoint_step: int | None = None, checkpoint_dir: str | None = None):
    """Main evaluation function."""
    
    logging.info(f"Running evaluation on: {platform.node()}")
    
    # Determine checkpoint directory
    checkpoint_path = checkpoint_dir if checkpoint_dir is not None else str(config.checkpoint_dir)
    
    # Create output directory
    eval_output_dir = epath.Path(checkpoint_path) / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trained policy
    logging.info("Loading trained policy...")
    policy = load_trained_policy(config, checkpoint_path, checkpoint_step)
    logging.info(f"Loaded policy successfully")
    
    # Create data loader with real data
    logging.info("Creating data loader...")
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    
    # Force video backend to pyav in config
    if hasattr(config.data, 'video_backend'):
        config.data.video_backend = "pyav"
    
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=False,  # Don't shuffle for evaluation
        num_batches=num_eval_batches,
        skip_norm_stats=True,  # Skip normalization stats - use those from the trained model
    )
    
    # Setup evaluation
    rng = jax.random.key(config.seed)
    all_metrics = []
    all_actions_gt = []
    all_actions_pred = []
    
    # Run evaluation
    logging.info(f"Running evaluation on {num_eval_batches} batches...")
    data_iter = iter(data_loader)
    
    for batch_idx in tqdm.tqdm(range(num_eval_batches), desc="Evaluating"):
        # Get batch
        observation, actions_gt = next(data_iter)
        
        # Run inference
        eval_rng = jax.random.fold_in(rng, batch_idx)
        actions_pred, metrics = evaluate_policy(policy, observation, actions_gt, eval_rng)
        
        # Store results
        all_metrics.append(metrics)
        all_actions_gt.append(np.array(actions_gt))
        all_actions_pred.append(np.array(actions_pred))
        
        # Log progress
        if batch_idx % 5 == 0:
            logging.info(f"Batch {batch_idx}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

    # Concatenate all results
    all_actions_gt = np.concatenate(all_actions_gt, axis=0)
    all_actions_pred = np.concatenate(all_actions_pred, axis=0)
    
    logging.info(f"Evaluation complete. Total samples: {all_actions_gt.shape[0]}")
    logging.info(f"Action shape: {all_actions_gt.shape}")
    
    # Create visualizations
    logging.info("Creating visualizations...")
    
    # Determine action names based on the dataset
    action_names = None
    data_config = config.data.create(config.assets_dirs, config.model)
    if hasattr(config.data, 'repo_id') and config.data.repo_id:
        if 'yam' in config.data.repo_id.lower():
            action_names = [
                'L_Joint_0', 'L_Joint_1', 'L_Joint_2', 'L_Joint_3', 'L_Joint_4', 'L_Joint_5', 'L_Gripper',
                'R_Joint_0', 'R_Joint_1', 'R_Joint_2', 'R_Joint_3', 'R_Joint_4', 'R_Joint_5', 'R_Gripper'
            ]
        elif 'xmi' in config.data.repo_id.lower():
            action_names = [
                'L_Rot_0', 'L_Rot_1', 'L_Rot_2', 'L_Rot_3', 'L_Rot_4', 'L_Rot_5',
                'L_Pos_X', 'L_Pos_Y', 'L_Pos_Z', 'L_Gripper',
                'R_Rot_0', 'R_Rot_1', 'R_Rot_2', 'R_Rot_3', 'R_Rot_4', 'R_Rot_5',
                'R_Pos_X', 'R_Pos_Y', 'R_Pos_Z', 'R_Gripper'
            ]
        elif 'aloha' in config.data.repo_id.lower():
            action_names = [
                'L_Joint_0', 'L_Joint_1', 'L_Joint_2', 'L_Joint_3', 'L_Joint_4', 'L_Joint_5', 'L_Gripper',
                'R_Joint_0', 'R_Joint_1', 'R_Joint_2', 'R_Joint_3', 'R_Joint_4', 'R_Joint_5', 'R_Gripper'
            ]
    
    # Action comparison plots
    plot_action_comparison(
        all_actions_gt,
        all_actions_pred,
        str(eval_output_dir / "action_comparison.png"),
        action_names=action_names,
    )
    
    # Error heatmap
    plot_error_heatmap(
        all_actions_gt,
        all_actions_pred,
        str(eval_output_dir / "error_heatmap.png"),
        action_names=action_names,
    )
    
    # Error distribution
    plot_error_distribution(
        all_actions_gt,
        all_actions_pred,
        str(eval_output_dir / "error_distribution.png"),
    )
    
    # Summary metrics
    summary_metrics = create_summary_metrics_plot(
        all_metrics,
        str(eval_output_dir / "summary_metrics.png"),
    )
    
    # Save detailed results (excluding config which may not be JSON serializable)
    results = {
        'summary_metrics': summary_metrics,
        'all_metrics': all_metrics,
        'evaluation_info': {
            'num_batches': num_eval_batches,
            'total_samples': int(all_actions_gt.shape[0]),
            'action_shape': [int(x) for x in all_actions_gt.shape],
            'checkpoint_step': checkpoint_step if checkpoint_step is not None else 'latest',
            'config_name': config.name if hasattr(config, 'name') else 'unknown',
        }
    }
    
    import json
    with open(eval_output_dir / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logging.info("=" * 60)
    logging.info("EVALUATION SUMMARY")
    logging.info("=" * 60)
    for metric_name, metric_data in summary_metrics.items():
        logging.info(f"{metric_name.upper()}: {metric_data['mean']:.6f} ± {metric_data['std']:.6f}")
    logging.info("=" * 60)
    logging.info(f"Results saved to: {eval_output_dir}")
    logging.info("Generated plots:")
    logging.info(f"  - Action comparison: {eval_output_dir / 'action_comparison.png'}")
    logging.info(f"  - Error heatmap: {eval_output_dir / 'error_heatmap.png'}")
    logging.info(f"  - Error distribution: {eval_output_dir / 'error_distribution.png'}")
    logging.info(f"  - Summary metrics: {eval_output_dir / 'summary_metrics.png'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a trained OpenPI model")
    parser.add_argument("--config-name", type=str, required=True, help="Name of the training config to use")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name (required if not set in config)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint directory (default: use config.checkpoint_dir)")
    parser.add_argument("--num-eval-batches", type=int, default=10, help="Number of batches to evaluate on")
    parser.add_argument("--checkpoint-step", type=int, default=None, help="Specific checkpoint step to load (default: latest)")
    
    args = parser.parse_args()
    
    # Initialize logging early for error messages
    init_logging()
    
    # Load config
    config = _config.get_config(args.config_name)
    
    # Set experiment name if provided
    if args.exp_name is not None:
        config = dataclasses.replace(config, exp_name=args.exp_name)
    
    # Validate that exp_name is set
    try:
        # This will trigger the error if exp_name is missing
        _ = config.checkpoint_dir
    except (TypeError, AttributeError) as e:
        if "PropagatingMissingType" in str(e) or "unsupported operand type" in str(e):
            logging.error("Experiment name (exp_name) is not set in the config and not provided via --exp-name")
            logging.error("Please provide --exp-name argument or ensure the config has exp_name set")
            exit(1)
        else:
            raise
    
    # Run evaluation
    main(config, args.num_eval_batches, args.checkpoint_step, args.checkpoint_dir) 