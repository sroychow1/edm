import torch
from training import networks

def main():
    torch.set_grad_enabled(False)
    
    # We will hook into the layers of RowSongUNet to ensure the 
    # output shape of all RowConv2d and RowUNetBlock maintain H=1.
    
    net = networks.RowSongUNet(
        img_resolution=64,
        in_channels=3,
        out_channels=3,
    )
    
    def hook_fn(module, input, output):
        if output.shape[2] != 1:
            print(f"FAILED: {module.__class__.__name__} produced shape {tuple(output.shape)}")
            
    for name, module in net.named_modules():
        if isinstance(module, (networks.RowConv2d, networks.RowUNetBlock)):
            module.register_forward_hook(hook_fn)
            
    x = torch.randn(2, 3, 1, 64)
    noise_labels = torch.randn(2)
    
    print("Running forward pass...")
    out = net(x, noise_labels, class_labels=None)
    
    if out.shape[2] == 1:
        print("SUCCESS: Final output has H=1")
        print(f"Final shape: {tuple(out.shape)}")
    else:
        print(f"FAILED: Final output shape: {tuple(out.shape)}")

if __name__ == '__main__':
    main()
