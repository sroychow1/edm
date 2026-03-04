import torch
from training.networks import EDMPrecond
from torch.optim import Adam
import dnnlib

def generate(net, latents, class_labels=None, randn_like=torch.randn_like,
             num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
             S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next

def train_manual(net, target, optimizer, iterations=6000):
    # EDM Loss explicit logic for a batch of 1
    P_mean=-1.2; P_std=1.2; sigma_data=0.5
    
    net.train()
    for step in range(iterations):
        optimizer.zero_grad()
        
        # Draw noise
        rnd_normal = torch.randn([1, 1, 1, 1], device=target.device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        
        weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
        
        n = torch.randn_like(target) * sigma
        D_yn = net(target + n, sigma)
        
        loss = weight * ((D_yn - target) ** 2)
        loss = loss.mean()
        
        loss.backward()
        optimizer.step()
        
        if (step+1) % 500 == 0:
            print(f"Iter {step+1:4d}, Loss: {loss.item():.6f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    H, W = 1, 64
    C = 3
    
    # Simple Target: Linspace in each channel
    row = torch.linspace(-1, 1, W).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1) # [3, 1, 64]
    
    # Overfit to target exactly
    target = row.unsqueeze(0).to(device)
    
    net = EDMPrecond(
        img_resolution=W,
        img_channels=C,
        model_type='RowSongUNet',
        use_fp16=False,
    ).to(device)
    
    from training.networks import weight_init
    # Make sure we're training fresh
    optimizer = Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.99))
    
    print("\nTraining Manual EDM Loss...")
    train_manual(net, target, optimizer, iterations=7000)
    
    print("\nSampling from overtrained model...")
    net.eval()
    with torch.no_grad():
        latents = torch.randn([1, C, H, W], device=device)
        # Use more steps for better sample fidelity
        generated = generate(net, latents, num_steps=50)
    
    mse = torch.nn.functional.mse_loss(generated.to(torch.float32), target).item()
    print(f"Mean Squared Error: {mse:.8f}")
    
    if mse < 1e-2:
        print("\033[92mSUCCESS: Overfit perfectly!\033[0m")
    else:
        print("\033[91mFAILED: Could not overfit. Output varies.\033[0m")

if __name__ == '__main__':
    main()
