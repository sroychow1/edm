"""Lightweight overfitting monitor for EDM training runs.

Evaluates saved snapshots from a training run directory and produces:
  - Visual sample grids per snapshot
  - Nearest-neighbor distance to training data (memorization check)
  - Sample diversity (mode collapse check)
  - Training loss curve from stats.jsonl
  - Summary CSV and plots

Usage:
    CUDA_VISIBLE_DEVICES=2 python monitor_training.py \\
        --run-dir=training-runs/00000-... \\
        --data=simulation/ball_sim_edm_z_sweep.zip \\
        --num-samples=64 --nn-subset=2048 --every=1
"""

import csv
import glob
import json
import math
import os
import pickle
import re
import sys

import click
import numpy as np
import torch
import PIL.Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate import edm_sampler
from training.dataset import ImageFolderDataset
import dnnlib
import torch_utils.persistence  # noqa: F401 – registers unpickle hooks


def load_training_subset(data_path, nn_subset, device):
    """Load a random subset of training images, normalized to [-1, 1]."""
    dataset = ImageFolderDataset(path=data_path, max_size=nn_subset, random_seed=0)
    images = []
    for i in range(len(dataset)):
        img, _label = dataset[i]
        images.append(torch.from_numpy(img).to(torch.float32))
    images = torch.stack(images).to(device) / 127.5 - 1  # uint8 CHW → [-1, 1]
    dataset.close()
    return images


def generate_samples(net, num_samples, device, seed=0):
    """Generate samples using edm_sampler with deterministic seeds."""
    latents = torch.randn(
        [num_samples, net.img_channels, net.img_resolution, net.img_resolution],
        device=device, generator=torch.Generator(device=device).manual_seed(seed),
    )
    with torch.no_grad():
        images = edm_sampler(net, latents)
    return images.to(torch.float32)


def make_grid(images_nchw, ncols=8):
    """Stitch a batch of CHW uint8 images into a single grid image."""
    n, c, h, w = images_nchw.shape
    nrows = math.ceil(n / ncols)
    pad = nrows * ncols - n
    if pad > 0:
        images_nchw = torch.cat([images_nchw, torch.zeros(pad, c, h, w, dtype=images_nchw.dtype)])
    grid = images_nchw.reshape(nrows, ncols, c, h, w)
    grid = grid.permute(2, 0, 3, 1, 4).reshape(c, nrows * h, ncols * w)
    return grid


def compute_nn_distances(generated, train_subset):
    """Per-generated-image minimum L2 distance to training subset.

    Returns (nn_min across batch, nn_mean across batch).
    Both operate on flattened pixel vectors.
    """
    gen_flat = generated.reshape(generated.shape[0], -1)
    train_flat = train_subset.reshape(train_subset.shape[0], -1)

    # Compute pairwise L2²  via expansion: ||a-b||² = ||a||² + ||b||² - 2a·b
    gen_sq = (gen_flat ** 2).sum(dim=1, keepdim=True)
    train_sq = (train_flat ** 2).sum(dim=1, keepdim=True)
    dists_sq = gen_sq + train_sq.T - 2 * gen_flat @ train_flat.T  # [N_gen, N_train]
    dists_sq = dists_sq.clamp(min=0)

    nn_per_sample = dists_sq.min(dim=1).values.sqrt()  # closest train image per generated sample
    return float(nn_per_sample.min()), float(nn_per_sample.mean())


def compute_diversity(generated):
    """Mean pairwise L2 distance among generated samples."""
    flat = generated.reshape(generated.shape[0], -1)
    sq = (flat ** 2).sum(dim=1, keepdim=True)
    dists_sq = sq + sq.T - 2 * flat @ flat.T
    dists_sq = dists_sq.clamp(min=0)

    n = flat.shape[0]
    mask = torch.triu(torch.ones(n, n, device=flat.device, dtype=torch.bool), diagonal=1)
    pairwise = dists_sq[mask].sqrt()
    return float(pairwise.mean())


def parse_stats_jsonl(run_dir):
    """Extract (kimg, loss_mean) pairs from stats.jsonl."""
    path = os.path.join(run_dir, 'stats.jsonl')
    records = []
    if not os.path.isfile(path):
        return records
    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            kimg = entry.get('Progress/kimg', {}).get('mean', None)
            loss = entry.get('Loss/loss', {}).get('mean', None)
            if kimg is not None and loss is not None:
                records.append((float(kimg), float(loss)))
    return records


def discover_snapshots(run_dir, every):
    """Find snapshot .pkl files sorted by kimg, taking every Nth."""
    pattern = os.path.join(run_dir, 'network-snapshot-*.pkl')
    paths = sorted(glob.glob(pattern))
    kimg_re = re.compile(r'network-snapshot-(\d+)\.pkl$')
    result = []
    for p in paths:
        m = kimg_re.search(p)
        if m:
            result.append((int(m.group(1)), p))
    return result[::every]


def save_plots(monitor_dir, loss_records, snapshot_metrics):
    """Generate and save summary matplotlib plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if loss_records:
        kimgs, losses = zip(*loss_records)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(kimgs, losses, linewidth=0.8)
        ax.set_xlabel('kimg')
        ax.set_ylabel('Loss (mean)')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(monitor_dir, 'loss_curve.png'), dpi=150)
        plt.close(fig)

    if not snapshot_metrics:
        return

    s_kimgs = [m['kimg'] for m in snapshot_metrics]
    nn_mins = [m['nn_min'] for m in snapshot_metrics]
    nn_means = [m['nn_mean'] for m in snapshot_metrics]
    diversities = [m['diversity'] for m in snapshot_metrics]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s_kimgs, nn_mins, label='nn_min', marker='.', markersize=3, linewidth=0.8)
    ax.plot(s_kimgs, nn_means, label='nn_mean', marker='.', markersize=3, linewidth=0.8)
    ax.set_xlabel('kimg')
    ax.set_ylabel('L2 distance (pixel space, [-1,1])')
    ax.set_title('Nearest-Neighbor Distance to Training Set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(monitor_dir, 'nn_distance.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s_kimgs, diversities, marker='.', markersize=3, linewidth=0.8)
    ax.set_xlabel('kimg')
    ax.set_ylabel('Mean pairwise L2 distance')
    ax.set_title('Sample Diversity')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(monitor_dir, 'diversity.png'), dpi=150)
    plt.close(fig)


@click.command()
@click.option('--run-dir', help='Training run directory', metavar='DIR', type=str, required=True)
@click.option('--data', help='Path to training dataset zip/dir', metavar='PATH', type=str, required=True)
@click.option('--num-samples', help='Samples to generate per snapshot', metavar='INT', type=int, default=64, show_default=True)
@click.option('--nn-subset', help='Training images for NN check', metavar='INT', type=int, default=2048, show_default=True)
@click.option('--every', help='Evaluate every Nth snapshot', metavar='INT', type=int, default=1, show_default=True)
@click.option('--device', help='Torch device', metavar='STR', type=str, default='cuda:0', show_default=True)
def main(run_dir, data, num_samples, nn_subset, every, device):
    """Evaluate EDM training snapshots for overfitting signals."""
    device = torch.device(device)
    monitor_dir = os.path.join(run_dir, 'monitor')
    grid_dir = os.path.join(monitor_dir, 'grids')
    os.makedirs(grid_dir, exist_ok=True)

    # Load training subset once.
    print(f'Loading {nn_subset} training images from "{data}"...')
    train_images = load_training_subset(data, nn_subset, device)
    print(f'  Loaded {train_images.shape[0]} images, shape {list(train_images.shape[1:])}')

    # Discover snapshots.
    snapshots = discover_snapshots(run_dir, every)
    if len(snapshots) == 0:
        print('No snapshots found. Exiting.')
        return
    print(f'Found {len(snapshots)} snapshot(s) to evaluate.')

    # Evaluate each snapshot.
    snapshot_metrics = []
    for idx, (kimg, pkl_path) in enumerate(snapshots):
        print(f'\n[{idx+1}/{len(snapshots)}] Snapshot at {kimg} kimg: {os.path.basename(pkl_path)}')

        with open(pkl_path, 'rb') as f:
            net = pickle.load(f)['ema'].to(device)
        net.eval()

        samples = generate_samples(net, num_samples, device)

        # Visual grid.
        grid_uint8 = (samples * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu()
        grid_img = make_grid(grid_uint8)
        if grid_img.shape[0] == 1:
            pil_img = PIL.Image.fromarray(grid_img[0].numpy(), 'L')
        else:
            pil_img = PIL.Image.fromarray(grid_img.permute(1, 2, 0).numpy(), 'RGB')
        grid_path = os.path.join(grid_dir, f'grid_{kimg:06d}.png')
        pil_img.save(grid_path)
        print(f'  Grid saved: {grid_path}')

        # NN distance.
        nn_min, nn_mean = compute_nn_distances(samples.to(device), train_images)
        print(f'  NN distance — min: {nn_min:.4f}, mean: {nn_mean:.4f}')

        # Diversity.
        diversity = compute_diversity(samples.to(device))
        print(f'  Diversity (mean pairwise L2): {diversity:.4f}')

        snapshot_metrics.append(dict(kimg=kimg, nn_min=nn_min, nn_mean=nn_mean, diversity=diversity))

        del net
        torch.cuda.empty_cache()

    # Parse training loss.
    loss_records = parse_stats_jsonl(run_dir)
    print(f'\nParsed {len(loss_records)} loss entries from stats.jsonl.')

    # Write metrics CSV.
    csv_path = os.path.join(monitor_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['kimg', 'nn_min', 'nn_mean', 'diversity'])
        writer.writeheader()
        writer.writerows(snapshot_metrics)
    print(f'Metrics CSV saved: {csv_path}')

    # Generate plots.
    save_plots(monitor_dir, loss_records, snapshot_metrics)
    print(f'Plots saved to: {monitor_dir}')

    # Summary table.
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'{"kimg":>8s}  {"nn_min":>10s}  {"nn_mean":>10s}  {"diversity":>10s}')
    print('-' * 70)
    for m in snapshot_metrics:
        print(f'{m["kimg"]:8d}  {m["nn_min"]:10.4f}  {m["nn_mean"]:10.4f}  {m["diversity"]:10.4f}')
    print('-' * 70)
    print()
    print('Interpretation:')
    print('  nn_min → 0      : model is reproducing training images (memorization)')
    print('  nn_mean falling  : model drifting toward training set (overfitting)')
    print('  diversity falling: mode collapse (repetitive samples)')
    print()


if __name__ == '__main__':
    main()
