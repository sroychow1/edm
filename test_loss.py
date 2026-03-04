import torch
from training.loss import EDMLoss
from training.networks import EDMPrecond

# Let's see what EDMLoss actually does 
loss_fn = EDMLoss()
net = EDMPrecond(img_resolution=64, img_channels=3, model_type='RowSongUNet', use_fp16=False)
target = torch.randn(1, 3, 1, 64)
# Look at loss_fn signature: forward(self, net, images, labels=None, augment_pipe=None)
loss = loss_fn(net, target, labels=None)
print("Loss shape:", loss.shape)

