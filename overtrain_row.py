import torch
import torch.nn.functional as F
from training.networks import EDMPrecond
from torch.optim import Adam
import numpy as np

def generate(net, latents, num_steps=50, rho=7):
    # This is standard EDM deterministic sampling (Euler) from EDM paper
    sigma_min = max(0.002, net.sigma_min)
    sigma_max = min(80, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        t_hat = net.round_sigma(t_cur)
        x_hat = x_cur
        
        denoised = net(x_hat, t_hat, None).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        
        # Heun step
        if i < num_steps - 1:
            denoised = net(x_next, t_next, None).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next.to(torch.float32)

def main():
    print("Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Target: 1D signal of shape (1, 3, 1, 64)
    target = torch.linspace(-1, 1, 64).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
    
    print("Setting up RowSongUNet...")
    net = EDMPrecond(img_resolution=64, img_channels=3, model_type='RowSongUNet').to(device)
    
    # Lower learning rate to prevent exploding loss like last time
    optimizer = Adam(net.parameters(), lr=2e-4) # from 1e-3
    
    print("Overtraining on a single row image...")
    epochs = 4000
    net.train()
    
    for step in range(epochs):
        optimizer.zero_grad()
        
        # Standard EDM Loss Noise sampling
        rnd_normal = torch.randn([1, 1, 1, 1], device=target.device)
        sigma = (rnd_normal * 1.2 - 1.2).exp()
        weight = (sigma ** 2 + 0.5 ** 2) / (sigma * 0.5) ** 2
        
        # Add noise
        n = torch.randn_like(target) * sigma
        D_yn = net(target + n, sigma)
        
        # L2 Loss
        loss = (weight * ((D_yn - target) ** 2)).mean()
        loss.backward()
        
        # Gradient clipping to help stability
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        
        if (step+1) % 500 == 0: 
            print(f"Step {step+1:4d}, Loss: {loss.item():.6f}")

    print("Training Complete. Sampling...")
    net.eval()
    
    # Run multiple samples to ensure stability
    mses = []
    with torch.no_grad():
        for i in range(5):
            latents = torch.randn([1, 3, 1, 64], device=device)
            # Use plenty of Euler steps
            generated = generate(net, latents, num_steps=100)
            mse = F.mse_loss(generated, target).item()
            mses.append(mse)

    avg_mse = sum(mses)/len(mses)
    print(f"\nAverage Mean Squared Error over 5 samples: {avg_mse:.6f}")
    
    if avg_mse < 0.05:
         print("\n\033[92mSUCCESS: The RowSongUNet memorized and generated the target row!\033[0m")
    else:
         print("\n\033[91mFAILED: Could not overfit perfectly. Validation MSE is relatively high.\033[0m")

if __name__ == '__main__':
    main()
