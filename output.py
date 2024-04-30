# -*- coding: utf-8 -*-

# Local application imports
from train import MyDDPM, MyUNet, save_images, generate_new_images, torch, get_config, device

# Getting configuration
config = get_config()

# Destructuring the dictionary into variables
model_path = config['model']
img_size = config['img_size']
channel = config['channel']
n_steps = config['n_steps']
min_beta = config['min_beta']
max_beta = config['max_beta']
n_images = config['n_images']

# Loading the trained model
model = MyDDPM(MyUNet(channel, img_size), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded")

# Generate new images
print("Generating new images")
generated = generate_new_images(
        model,
        n_samples=100,
        device=device,
        gif_name=None,
        gif=False
    )

# Show and save images
save_images(generated, n_images=n_images)