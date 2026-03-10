"""Train RowSongUNet on all rows of all images in a directory.

Uses ImageFolderRowDataset to iterate over every row of every image in the
given directory.  After training, logs a figure with 25 random ground-truth
rows vs 25 generated rows (line plots, x = pixel index, y in [-1, 1]).

No EMA, no LR warmup -- constant learning rate throughout.

Usage:
    torchrun --standalone --nproc_per_node=1 train_all_images_rows_pipeline.py \
        --dir datasets/peristalsis-256x256 --outdir training-runs
"""

import os
import io
import copy
import json
import pickle
import time

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch

import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc
from training.dataset import ImageFolderRowDataset
from torch.utils.tensorboard import SummaryWriter
from tensorboard.compat.proto.summary_pb2 import Summary

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides')

# ---------------------------------------------------------------------------

def edm_sampler(net, latents, num_steps=100, sigma_min=0.002, sigma_max=80, rho=7):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_hat = x_next
        t_hat = net.round_sigma(t_cur)
        denoised = net(x_hat, t_hat, None).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if i < num_steps - 1:
            denoised = net(x_next, t_next, None).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next.to(torch.float32)


def log_figure(writer, tag, fig, step):
    """Write a matplotlib figure to TensorBoard bypassing the broken PIL resize."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    fig_image = PIL.Image.open(buf).convert('RGB')
    fig_arr = np.array(fig_image)
    img_buf = io.BytesIO()
    fig_image.save(img_buf, format='PNG')
    img_summary = Summary.Value(
        tag=tag,
        image=Summary.Image(
            height=fig_arr.shape[0],
            width=fig_arr.shape[1],
            colorspace=3,
            encoded_image_string=img_buf.getvalue(),
        ),
    )
    writer.file_writer.add_summary(Summary(value=[img_summary]), step)
    img_buf.close()
    buf.close()
    plt.close(fig)


def make_grid_figure(gt_rows, gen_rows, row_indices):
    """Build a 5x5 grid of line plots comparing GT vs generated rows."""
    n = len(row_indices)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows), squeeze=False)
    positions = np.arange(gt_rows.shape[-1])

    for i in range(n):
        ax = axes[i // cols, i % cols]
        ax.plot(positions, gt_rows[i, 0, 0], label='GT', linewidth=1.0)
        ax.plot(positions, gen_rows[i, 0, 0], label='Generated', linewidth=1.0, linestyle='--')
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f'row {row_indices[i]}', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=6)

    for i in range(n, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    fig.suptitle('Ground Truth vs Generated rows', fontsize=14)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------

@click.command()
@click.option('--dir',          'data_dir', help='Path to image directory', type=str, required=True)
@click.option('--outdir',       help='Where to save results',              type=str, default='training-runs')
@click.option('--steps',        help='Number of training steps',           type=int, default=4000)
@click.option('--batch',        help='Batch size',                         type=int, default=32)
@click.option('--lr',           help='Learning rate',                      type=float, default=2e-4)
@click.option('--seed',         help='Random seed',                        type=int, default=0)
@click.option('--log-every',    'log_every', help='Log every N steps',     type=int, default=500)
@click.option('--sample-every', 'sample_every', help='Sample every N steps', type=int, default=500)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    device = torch.device('cuda')

    np.random.seed(opts.seed)
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = True

    sigma_data = 0.5
    P_mean, P_std = -1.2, 1.2

    # --- Dataset + DataLoader ---------------------------------------------
    dataset = ImageFolderRowDataset(path=opts.data_dir, cache=True)
    C = dataset.num_channels
    W = dataset.row_resolution
    num_rows = len(dataset)
    num_images = dataset.num_images
    print(f'Directory: {opts.data_dir}  ({num_images} images, {C}ch, {num_rows} total rows, width {W})')

    sampler = misc.InfiniteSampler(dataset=dataset, rank=0, num_replicas=1, seed=opts.seed)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=opts.batch)
    data_iter = iter(dataloader)

    # --- Run directory ----------------------------------------------------
    dir_name = os.path.basename(opts.data_dir.rstrip('/\\')) or 'images'
    desc = f'{dir_name}-allrows-fp32'
    prev_run_dirs = []
    if os.path.isdir(opts.outdir):
        prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
    prev_run_ids = [int(x.split('-')[0]) for x in prev_run_dirs if x.split('-')[0].isdigit()]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
        json.dump(dict(opts), f, indent=2)
    print(f'Output directory: {run_dir}')

    # --- Network ----------------------------------------------------------
    net = dnnlib.util.construct_class_by_name(
        class_name='training.networks.EDMPrecond',
        img_resolution=W, img_channels=C, label_dim=0,
        model_type='RowSongUNet',
        embedding_type='positional', encoder_type='standard', decoder_type='standard',
        channel_mult_noise=1, resample_filter=[1, 1],
        model_channels=128, channel_mult=[1, 2, 2, 2],
        dropout=0.0, use_fp16=False,
    )
    net.train().requires_grad_(True).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-8)

    # --- TensorBoard ------------------------------------------------------
    tb_dir = os.path.join(run_dir, 'tb')
    writer = SummaryWriter(log_dir=tb_dir)
    print(f'TensorBoard logs: {tb_dir}')

    # --- Training loop ----------------------------------------------------
    print(f'Training for {opts.steps} steps on {num_rows} rows from {num_images} images ...')
    start_time = time.time()

    for step in range(opts.steps):
        images, _labels = next(data_iter)
        images = images.to(device).to(torch.float32) / 127.5 - 1

        optimizer.zero_grad()

        B = images.shape[0]
        rnd_normal = torch.randn([B, 1, 1, 1], device=device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2

        n = torch.randn_like(images) * sigma
        D_yn = net(images + n, sigma)
        loss = (weight * ((D_yn - images) ** 2)).mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        # -- Logging -------------------------------------------------------
        if (step + 1) % opts.log_every == 0 or step == 0:
            loss_val = loss.item()
            elapsed = time.time() - start_time
            writer.add_scalar('Loss/train', loss_val, step)
            print(f'  step {step+1:5d}/{opts.steps}  loss {loss_val:.6f}  time {elapsed:.0f}s')

        # -- Sample and log ------------------------------------------------
        if (step + 1) % opts.sample_every == 0:
            net.eval()
            with torch.no_grad():
                latents = torch.randn([1, C, 1, W], device=device)
                generated = edm_sampler(net, latents, num_steps=100)
            net.train()
            writer.add_scalar('Sample/max', generated.max().item(), step)
            writer.add_scalar('Sample/min', generated.min().item(), step)

    # --- Post-training visualization --------------------------------------
    print('\nTraining complete. Generating post-training comparison ...')
    net.eval()

    rng = np.random.RandomState(opts.seed)
    viz_indices = rng.choice(num_rows, size=min(25, num_rows), replace=False)
    viz_indices.sort()

    gt_rows = []
    for idx in viz_indices:
        row_uint8, _label = dataset[idx]
        row_f32 = row_uint8.astype(np.float32) / 127.5 - 1
        gt_rows.append(row_f32)
    gt_rows = np.stack(gt_rows)

    with torch.no_grad():
        latents = torch.randn([len(viz_indices), C, 1, W], device=device)
        gen_rows = edm_sampler(net, latents, num_steps=100).cpu().numpy()

    fig = make_grid_figure(gt_rows, gen_rows, viz_indices)
    log_figure(writer, 'Comparison/gt_vs_generated', fig, opts.steps)
    print(f'Logged 25-row comparison figure to TensorBoard.')

    # -- Save snapshot -----------------------------------------------------
    snap_path = os.path.join(run_dir, 'network-final.pkl')
    with open(snap_path, 'wb') as f:
        pickle.dump(dict(net=copy.deepcopy(net).cpu()), f)
    print(f'Saved final network to {snap_path}')

    writer.close()
    print('Done.')


if __name__ == '__main__':
    main()
