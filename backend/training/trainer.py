
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import FolderImageDataset
from models.generator import Generator
from models.discriminator import Discriminator
from models.perceptual import VGGPerceptualLoss
from training.metrics import calculate_psnr, calculate_ssim
from training.visualize import show_sample
from config import BATCH_SIZE, EPOCHS, PATIENCE, DEVICE, DATASET_PATH, OUTPUT_DIR

def train_srgan(dataset_path):
    dataset = FolderImageDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(DEVICE=="cuda"))

    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    perceptual_loss = VGGPerceptualLoss()
    bce = nn.BCELoss()
    g_opt = torch.optim.Adam(generator.parameters(), lr=1e-4)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    best_psnr = 0.0
    epochs_without_improvement = 0
    best_generator_state = None

    for epoch in range(EPOCHS):
        print(f"\n🎯 Epoch {epoch+1}/{EPOCHS}")
        generator.train()
        discriminator.train()

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        num_batches = 0

        for step, (lr, hr) in enumerate(dataloader):
            lr, hr = lr.to(DEVICE, non_blocking=True), hr.to(DEVICE, non_blocking=True)

            with torch.no_grad():
                fake_detached = generator(lr).detach()
            real_logits = discriminator(hr)
            fake_logits = discriminator(fake_detached)
            d_loss_real = bce(real_logits, torch.ones_like(real_logits))
            d_loss_fake = bce(fake_logits, torch.zeros_like(fake_logits))
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            fake = generator(lr)
            fake_logits_for_g = discriminator(fake)
            pixel_loss = torch.mean(torch.abs(hr - fake))
            perc_loss = perceptual_loss(fake, hr)
            adv_loss = bce(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
            g_loss = pixel_loss + 0.006 * perc_loss + 1e-3 * adv_loss
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            with torch.no_grad():
                psnr_batch = calculate_psnr(fake[0].cpu(), hr[0].cpu())
                ssim_batch = calculate_ssim(fake[0].cpu(), hr[0].cpu())

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_psnr += psnr_batch
            epoch_ssim += ssim_batch
            num_batches += 1

            if step % 20 == 0:
                print(f"Step {step}: D Loss = {d_loss.item():.4f}, G Loss = {g_loss.item():.4f}, PSNR = {psnr_batch:.2f} dB, SSIM = {ssim_batch:.4f}")

        avg_d_loss = epoch_d_loss / max(1, num_batches)
        avg_g_loss = epoch_g_loss / max(1, num_batches)
        avg_psnr = epoch_psnr / max(1, num_batches)
        avg_ssim = epoch_ssim / max(1, num_batches)

        print(f"\nEpoch Summary:")
        print(f"Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")
        print(f"Avg PSNR: {avg_psnr:.2f} dB, Avg SSIM: {avg_ssim:.4f}")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            epochs_without_improvement = 0
            best_generator_state = generator.state_dict()
            print(f"🔥 New best PSNR: {best_psnr:.2f} dB")
        else:
            epochs_without_improvement += 1
            print(f"⏳ No improvement for {epochs_without_improvement}/{PATIENCE} epochs")
            if epochs_without_improvement >= PATIENCE:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break

        if (epoch + 1) % 5 == 0:
            try:
                show_sample(generator, dataloader)
            except Exception as e:
                print("Visualization skipped:", e)

    print("\n✅ Training completed!")
    if best_generator_state is not None:
        torch.save(best_generator_state, os.path.join(OUTPUT_DIR, "generator_best.pth"))
        print("✅ Best generator saved: generator_best.pth in Drive")

    print("\nFinal Evaluation:")
    generator.eval()
    with torch.no_grad():
        test_psnr = 0.0
        test_ssim = 0.0
        test_samples = min(10, len(dataset))
        for i in range(test_samples):
            lr, hr = dataset[i]
            lr, hr = lr.unsqueeze(0).to(DEVICE), hr.unsqueeze(0).to(DEVICE)
            sr = generator(lr)
            test_psnr += calculate_psnr(sr[0].cpu(), hr[0].cpu())
            test_ssim += calculate_ssim(sr[0].cpu(), hr[0].cpu())
        print(f"Test PSNR: {test_psnr / test_samples:.2f} dB")
        print(f"Test SSIM: {test_ssim / test_samples:.4f}")
