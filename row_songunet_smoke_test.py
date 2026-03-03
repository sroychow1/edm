#!/usr/bin/env python3
# Minimal forward-pass smoke test for RowSongUNet.

import torch

from training import networks


def main():
    torch.set_grad_enabled(False)

    N, C, L = 2, 3, 64
    x = torch.randn([N, C, 1, L], dtype=torch.float32)
    sigma = torch.full([N], 1.0, dtype=torch.float32)

    net = networks.VPPrecond(
        img_resolution=L,
        img_channels=C,
        model_type='RowSongUNet',
        use_fp16=False,
    )

    y = net(x, sigma)
    print('x.shape =', tuple(x.shape))
    print('y.shape =', tuple(y.shape))
    assert y.shape == x.shape, f'Unexpected output shape: got {tuple(y.shape)} expected {tuple(x.shape)}'


if __name__ == '__main__':
    main()

