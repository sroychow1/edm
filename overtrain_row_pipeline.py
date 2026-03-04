"""Overfit RowSongUNet on a single row (row 128) with TensorBoard visualization.

Trains on a fixed target row every step (like overtrain_row.py) with:
  - Scalar: training loss, sample-vs-target MSE
  - Figure: overlaid line plots of generated vs ground-truth pixel values

Usage:
    torchrun --standalone --nproc_per_node=1 overtrain_row_pipeline.py \
        --image /path/to/image.png --outdir training-runs
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
import torch.nn.functional as F

import dnnlib
from torch_utils import distributed as dist
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

# ---------------------------------------------------------------------------

def make_comparison_figure(generated_row, target_row, step):
    """Return a matplotlib Figure comparing generated vs target pixel values."""
    C, _, W = generated_row.shape
    positions = np.arange(W)
    fig, axes = plt.subplots(C, 1, figsize=(10, 3 * C), squeeze=False)
    for c in range(C):
        ax = axes[c, 0]
        ax.plot(positions, target_row[c, 0], label='Ground truth', linewidth=1.5)
        ax.plot(positions, generated_row[c, 0], label='Generated', linewidth=1.5, linestyle='--')
        ax.set_ylabel(f'Channel {c}')
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel('Pixel position')
    fig.suptitle(f'Generated vs Ground Truth row 128  (step {step})', fontsize=12)
    fig.tight_layout()
    return fig


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

# ---------------------------------------------------------------------------

ROW_IDX = 128

@click.command()
@click.option('--image',        help='Path to single PNG image',           type=str, required=True)
@click.option('--outdir',       help='Where to save results',              type=str, default='training-runs')
@click.option('--steps',        help='Number of training steps',           type=int, default=4000)
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

    # --- Load target row 128 from image -----------------------------------
    image = np.array(PIL.Image.open(opts.image))
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    image = image.transpose(2, 0, 1)  # CHW uint8
    C, H, W = image.shape
    assert ROW_IDX < H, f'Row index {ROW_IDX} out of range for image with {H} rows'

    target_uint8 = image[:, ROW_IDX:ROW_IDX+1, :]      # [C, 1, W] uint8
    target = torch.from_numpy(target_uint8.astype(np.float32) / 127.5 - 1).unsqueeze(0).to(device)  # [1, C, 1, W]
    target_np = target[0].cpu().numpy()                  # [C, 1, W] for plotting

    print(f'Image: {opts.image}  ({C}ch, {H}x{W})')
    print(f'Target: row {ROW_IDX}, shape {list(target.shape)}, range [{target.min():.2f}, {target.max():.2f}]')

    # --- Run directory ----------------------------------------------------
    dataset_name = os.path.splitext(os.path.basename(opts.image))[0]
    desc = f'{dataset_name}-row{ROW_IDX}-overfit-fp32'
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

    # --- Training loop (fixed target, no dataloader) ----------------------
    print(f'Training for {opts.steps} steps on row {ROW_IDX} ...')
    start_time = time.time()

    for step in range(opts.steps):
        optimizer.zero_grad()

        rnd_normal = torch.randn([1, 1, 1, 1], device=device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2

        n = torch.randn_like(target) * sigma
        D_yn = net(target + n, sigma)
        loss = (weight * ((D_yn - target) ** 2)).mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        # -- Logging -------------------------------------------------------
        if (step + 1) % opts.log_every == 0 or step == 0:
            loss_val = loss.item()
            elapsed = time.time() - start_time
            writer.add_scalar('Loss/train', loss_val, step)
            print(f'  step {step+1:5d}/{opts.steps}  loss {loss_val:.6f}  time {elapsed:.0f}s')

        # -- Sample and compare --------------------------------------------
        if (step + 1) % opts.sample_every == 0 or step == 0:
            net.eval()
            with torch.no_grad():
                latents = torch.randn([1, C, 1, W], device=device)
                generated = edm_sampler(net, latents, num_steps=100)
            net.train()

            gen_np = generated[0].cpu().numpy()
            mse = float(((gen_np - target_np) ** 2).mean())
            writer.add_scalar('MSE/sample_vs_target', mse, step)
            print(f'           sample MSE: {mse:.6f}')

            fig = make_comparison_figure(gen_np, target_np, step + 1)
            log_figure(writer, f'Comparison/row{ROW_IDX}', fig, step)

    # --- Final evaluation (5 samples, like overtrain_row.py) --------------
    print('\nTraining complete. Final evaluation (5 samples) ...')
    net.eval()
    mses = []
    with torch.no_grad():
        for i in range(5):
            latents = torch.randn([1, C, 1, W], device=device)
            generated = edm_sampler(net, latents, num_steps=100)
            mse = F.mse_loss(generated, target).item()
            mses.append(mse)
            print(f'  sample {i}: MSE = {mse:.6f}')

    avg_mse = sum(mses) / len(mses)
    writer.add_scalar('MSE/final_avg', avg_mse, opts.steps)
    print(f'\nAverage MSE over 5 samples: {avg_mse:.6f}')

    if avg_mse < 0.05:
        print('\033[92mSUCCESS: memorized row 128!\033[0m')
    else:
        print('\033[91mFAILED: MSE too high, did not memorize.\033[0m')

    # -- Final comparison figure -------------------------------------------
    gen_np = generated[0].cpu().numpy()
    fig = make_comparison_figure(gen_np, target_np, opts.steps)
    log_figure(writer, f'Comparison/row{ROW_IDX}_final', fig, opts.steps)

    # -- Save snapshot -----------------------------------------------------
    snap_path = os.path.join(run_dir, 'network-final.pkl')
    with open(snap_path, 'wb') as f:
        pickle.dump(dict(net=copy.deepcopy(net).cpu()), f)
    print(f'Saved final network to {snap_path}')

    writer.close()
    print('Done.')


if __name__ == '__main__':
    main()
