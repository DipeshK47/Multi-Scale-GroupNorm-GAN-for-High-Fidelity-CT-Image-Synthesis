import os
import numpy as np
import torch
import torch.nn.functional as F

# Adjust the imports to match your project structure.
from models.Model_HA_GAN_256 import Encoder, Sub_Encoder, Generator, Discriminator, Code_Discriminator

# Device configuration: use CUDA if available; otherwise, use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create output directory "step_op"
output_dir = "/content/step_op"
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# 1. Load your single .npy image
# --------------------------
npy_path = '/content/preprocessed.npy'
img_np = np.load(npy_path)
# Assume the npy file shape is (D, H, W); add batch and channel dims to get (1, 1, D, H, W)
img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(device)
print("Input image shape:", img_tensor.shape)

# --------------------------
# 2. Instantiate Models & Load Checkpoints
# --------------------------
encoder = Encoder().to(device)
sub_encoder = Sub_Encoder(latent_dim=1024).to(device)
generator = Generator(mode='eval', latent_dim=1024).to(device)
discriminator = Discriminator().to(device)
code_discriminator = Code_Discriminator(code_size=1024).to(device)

# Helper function to remove the "module." prefix from checkpoint keys.
def remove_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

# Load Encoder checkpoint.
encoder_ckpt = torch.load('/content/HA_GAN_run1/E_iter80000.pth', map_location=device)
encoder.load_state_dict(remove_module_prefix(encoder_ckpt['model']))
print("Loaded Encoder checkpoint.")

# Load Sub_Encoder checkpoint.
sub_encoder_ckpt = torch.load('/content/HA_GAN_run1/Sub_E_iter80000.pth', map_location=device)
sub_encoder.load_state_dict(remove_module_prefix(sub_encoder_ckpt['model']))
print("Loaded Sub_Encoder checkpoint.")

# Load Generator checkpoint.
generator_ckpt = torch.load('/content/HA_GAN_run1/G_iter80000.pth', map_location=device)
generator.load_state_dict(remove_module_prefix(generator_ckpt['model']))
print("Loaded Generator checkpoint.")

# Load Discriminator checkpoint.
discriminator_ckpt = torch.load('/content/HA_GAN_run1/D_iter80000.pth', map_location=device)
discriminator.load_state_dict(remove_module_prefix(discriminator_ckpt['model']))
print("Loaded Discriminator checkpoint.")

# Set all models to evaluation mode.
encoder.eval()
sub_encoder.eval()
generator.eval()
discriminator.eval()
code_discriminator.eval()

# --------------------------
# 3. Pass the image through each module and save outputs
# --------------------------
with torch.no_grad():
    # (a) Run through Encoder.
    encoder_out = encoder(img_tensor)
np.save(os.path.join(output_dir, 'output_encoder.npy'), encoder_out.cpu().numpy())
print("Saved output from Encoder.")

with torch.no_grad():
    # (b) Run through Sub_Encoder.
    sub_encoder_out = sub_encoder(encoder_out)
    # If sub_encoder_out is 1D, add a batch dimension.
    if sub_encoder_out.dim() == 1:
        sub_encoder_out = sub_encoder_out.unsqueeze(0)
np.save(os.path.join(output_dir, 'output_sub_encoder.npy'), sub_encoder_out.cpu().numpy())
print("Saved output from Sub_Encoder.")

with torch.no_grad():
    # (c) Prepare input for Code_Discriminator.
    # Pool the encoder output to collapse spatial dimensions.
    pooled_encoder = F.adaptive_avg_pool3d(encoder_out, (1, 1, 1))  # shape: [batch, channels, 1, 1, 1]
    latent_from_encoder = pooled_encoder.view(pooled_encoder.size(0), -1)  # shape: [batch, channels]
    # Map the pooled features (likely 64 channels) to 1024 dimensions.
    temp_mapping = torch.nn.Linear(latent_from_encoder.size(1), 1024).to(device)
    latent_for_disc = temp_mapping(latent_from_encoder)
    code_disc_out = code_discriminator(latent_for_disc)
np.save(os.path.join(output_dir, 'output_code_discriminator.npy'), code_disc_out.cpu().numpy())
print("Saved output from Code_Discriminator.")

with torch.no_grad():
    # (d) Generate an image using the Generator from the Encoder latent code.
    # The generator expects an input vector of size 1024.
    generated_img_from_encoder = generator(latent_for_disc, crop_idx=None)
np.save(os.path.join(output_dir, 'output_generator_from_encoder.npy'), generated_img_from_encoder.cpu().numpy())
print("Saved Generator output (from mapped Encoder latent code).")

with torch.no_grad():
    # (e) Generate an image using the Generator from Sub_Encoder output.
    # Ensure the Sub_Encoder output is 2D: [batch, latent_dim].
    if sub_encoder_out.dim() < 2:
        sub_encoder_vector = sub_encoder_out.unsqueeze(0)
    else:
        sub_encoder_vector = sub_encoder_out.view(sub_encoder_out.size(0), -1)
    # If the vector is not 1024-dimensional, map it.
    if sub_encoder_vector.size(1) != 1024:
        temp_mapping2 = torch.nn.Linear(sub_encoder_vector.size(1), 1024).to(device)
        sub_encoder_vector = temp_mapping2(sub_encoder_vector)
    generated_img_from_sub = generator(sub_encoder_vector, crop_idx=None)
np.save(os.path.join(output_dir, 'output_generator_from_sub_encoder.npy'), generated_img_from_sub.cpu().numpy())
print("Saved Generator output (from Sub_Encoder latent code).")

with torch.no_grad():
    # (f) Run through Discriminator.
    # The discriminator expects a high-resolution crop and a low-resolution full volume.
    img_lowres = F.interpolate(img_tensor, scale_factor=0.25, mode='trilinear', align_corners=False)
    # For a high-res crop, take a sub-volume along the depth.
    crop_size = img_tensor.size(2) // 8  # e.g., if depth is 256 then crop_size is 32.
    img_crop = img_tensor[:, :, 0:crop_size, :, :]
    disc_out = discriminator(img_crop, img_lowres, crop_idx=0)
if isinstance(disc_out, tuple):
    disc_logit, disc_class = disc_out
    np.save(os.path.join(output_dir, 'output_discriminator_logit.npy'), disc_logit.cpu().numpy())
    np.save(os.path.join(output_dir, 'output_discriminator_class.npy'), disc_class.cpu().numpy())
    print("Saved outputs from Discriminator (logit & class).")
else:
    np.save(os.path.join(output_dir, 'output_discriminator.npy'), disc_out.cpu().numpy())
    print("Saved output from Discriminator.")

print("All intermediate outputs have been saved in the folder:", output_dir)
