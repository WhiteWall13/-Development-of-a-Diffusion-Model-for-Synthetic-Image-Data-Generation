# -*- coding: utf-8 -*-
# TODO : Gerer les réseaux en fonction de la taille de l'image et des channels
# TODO : Gérer la seed
# TODO :

# Standard library imports
import os
import random
import configparser

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Lambda, Grayscale, Resize
import imageio
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import einops

# Local application imports
from unet import MyUNet


def get_config():
    """
    Reads the configuration from 'config.ini' and returns it as a dictionary.
    
    :return: A dictionary containing the configurations.
    :rtype: dict
    """
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the config.ini file
    config.read('config.ini')

    # Access the DEFAULT section and return it as a dictionary
    config_dict = {key: eval(value) if value in ['True', 'False', 'None'] else value for key, value in config['DEFAULT'].items()}
    
    # Convert numeric values from strings to appropriate types
    config_dict['img_size'] = int(config_dict['img_size'])
    config_dict['n_channel'] = int(config_dict['n_channel'])
    config_dict['batch_size'] = int(config_dict['batch_size'])
    config_dict['n_epochs'] = int(config_dict['n_epochs'])
    config_dict['n_steps'] = int(config_dict['n_steps'])
    config_dict['lr'] = float(config_dict['lr'])
    config_dict['min_beta'] = float(config_dict['min_beta'])
    config_dict['max_beta'] = float(config_dict['max_beta'])
    config_dict['debug'] = bool(config_dict['debug'])
    config_dict['n_images'] = int(config_dict['n_images'])
    
    return config_dict

# Getting configuration
config = get_config()

# Destructuring the dictionary into variables
dataset_path = config['dataset']
class_name = config['class']
model_path = config['model']
img_size = config['img_size']
channel = config['n_channel']
batch_size = config['batch_size']
n_epochs = config['n_epochs']
lr = config['lr']
n_steps = config['n_steps']
min_beta = config['min_beta']
max_beta = config['max_beta']
debug = config['debug']

# Checking if the arguments are valid
valid_img_sizes = {128, 256, 512}
valid_channels = {1, 3}

if img_size not in valid_img_sizes:
    raise ValueError(f"Invalid img_size: {img_size}. Must be one of {valid_img_sizes}.")

if channel not in valid_channels:
    raise ValueError(f"Invalid number of channel: {channel}. Must be one of {valid_channels}.")

# Definitions
img_path = "debug_img"

# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else ("cpu"))
print(f"Using device: {device} \t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))


def make_path(path):
    """
    A function that creates a directory at the specified path if it does not already exist.
    
    Args:
    - path (str): The path where the directory will be created.
    
    Returns:
    - None
    """
    if not os.path.exists(path):
        os.makedirs(path)
        

def show_images(images, title, path=img_path):
    """
    Show images in a grid layout with a given title and save the resulting figure as a PNG file.

    Parameters:
    - images (torch.Tensor or numpy.ndarray): The images to be displayed. If a torch.Tensor is provided, it will be converted to a numpy array.
    - title (str): The title of the figure.
    - path (str, optional): The path to save the figure. Defaults to img_path.

    Returns:
    None
    """
    # Create path if it does not exist
    make_path(path)

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)
    
    # Save the figure
    plt.savefig(f'{path}/{title}.png')
    print(f"Saved at {path}/{title}.png")
    plt.close()
    

def save_images(images, path="output", prefix="output", n_images=1):
    """
    Saves a specified number of images to a given path with a specified prefix.

    Parameters:
        images (torch.Tensor): The tensor containing the images to be saved.
        path (str, optional): The path to save the images. Defaults to "output".
        prefix (str, optional): The prefix to be added to the saved image filenames. Defaults to "output".
        n_images (int, optional): The number of images to be saved. Defaults to 1.

    Returns:
        None
    """
    # Create path if it does not exist
    make_path(path)
    
    # Get the number of images to save
    n_images = n_images if n_images is not None else images.size(0)
    
    # Save the images
    for i in range(min(n_images, images.size(0))):
        img = images[i].cpu().squeeze()  # De-tensorize
        img = (img - img.min()) / (img.max() - img.min()) * 255  # Normalization
        img = img.byte()  # Convert to uint8
        img_pil = Image.fromarray(img.numpy(), mode="L")  # Convert to image PIL
        img_pil.save(os.path.join(path, f"{prefix}_{i+1}.jpg"))
        print(f"Image {i+1} saved at {path}")
    

def show_first_batch(loader):
    """
    Shows the first batch of images.

    Parameters:
    - loader: The data loader containing the batches of images.

    Returns:
    None
    """
    make_path(f"{img_path}/original")
    for batch in loader:
        show_images(batch[0], "original/img_first_batch")
        break

class SquarePad:
    def __call__(self, image):
        """
        A function that pads the input image to make it square by adding padding to the left, right, top, and bottom.
        
        Args:
        - image: The input image to be padded.
        
        Returns:
        - Tensor: The padded image.
        """
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')

# DDPM class
class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=None, image_chw=(channel, img_size, img_size)):
        """
        Initializes a new instance of the MyDDPM class.

        Args:
            network (torch.nn.Module): The neural network model used for denoising.
            n_steps (int, optional): The number of steps in the beta schedule. Defaults to n_steps.
            min_beta (float, optional): The minimum value of the beta schedule. Defaults to min_beta.
            max_beta (float, optional): The maximum value of the beta schedule. Defaults to max_beta.
            device (str, optional): The device to run the computations on. Defaults to None.
            image_chw (tuple, optional): The shape of the input images. Defaults to (channel, img_size, img_size).

        Returns:
            None
        """
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        """
        Applies the forward pass of the DDPM model to generate noisy images.

        Args:
            x0 (torch.Tensor): The input image tensor of shape (n, c, h, w).
            t (int): The time step of the DDPM model.
            eta (torch.Tensor, optional): The noise tensor of shape (n, c, h, w). If not provided, a random noise tensor is generated.

        Returns:
            torch.Tensor: The noisy image tensor of shape (n, c, h, w).
        """
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t):
        """
        Run each image through the network for each timestep t in the vector t.
        The network returns its estimation of the noise that was added.

        Parameters:
            x (torch.Tensor): The input image tensor of shape (n, c, h, w).
            t (torch.Tensor): The time step tensor of shape (n,).

        Returns:
            torch.Tensor: The output tensor of shape (n, c, h, w).
        """
        return self.network(x, t)


def show_forward(ddpm, loader, device):
    """
    Shows the forward process.

    Parameters:
        ddpm: The ddpm model.
        loader: The data loader.
        device: The device for computation.

    Returns:
        None
    """
    make_path(f"{img_path}/original")
    make_path(f"{img_path}/noisy")
    for batch in loader:
        imgs = batch[0]
        show_images(imgs, "original/Original_images")
        for percent in [0, 0.25, 0.5, 0.75, 1]:
            t_index = int(percent * (ddpm.n_steps - 1))
            t_values = [t_index for _ in range(imgs.shape[0])]
            noisy_imgs = ddpm(imgs.to(device), t_values)
            show_images(noisy_imgs, f"noisy/DDPM_Noisy_images_{int(percent * 100)}%")
        break


def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=100, gif_name="sampling.gif", c=channel, h=img_size, w=img_size, gif=True):
    """
    Generates new images using the given DDPM model.

    Args:
        ddpm (MyDDPM): The DDPM model.
        n_samples (int, optional): The number of samples to generate. Defaults to 16.
        device (str, optional): The device for computation. Defaults to None.
        frames_per_gif (int, optional): The number of frames per GIF. Defaults to 100.
        gif_name (str, optional): The name of the GIF file. Defaults to "sampling.gif".
        c (int, optional): The number of channels in the images. Defaults to channel.
        h (int, optional): The height of the images. Defaults to img_size.
        w (int, optional): The width of the images. Defaults to img_size.
        gif (bool, optional): Whether to generate a GIF. Defaults to True.

    Returns:
        torch.Tensor: The generated images.

    Description:
        This function generates new images using the given DDPM model. It starts by initializing the images with random noise.
        Then, it iterates over the time steps in reverse order. For each time step, it estimates the noise to be removed and performs
        partial denoising of the images. If the time step is greater than 0, it adds some more noise to the images.
        After denoising, it adds frames to the GIF if the gif flag is set to True and the current index is in the frame indices or
        the time step is 0. The frames are stored in a list. Finally, if the gif flag is set to True, the GIF is stored as a file."""
     
    # Defining frame indices   
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []
    
    with torch.no_grad():
        # Defining device
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)
            
            # Estimating alpha_t and alpha_t_bar
            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            # Adding some noise like in Langevin Dynamics fashion
            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                x = x + sigma_t * z

            # Adding frames to the GIF
            if gif and idx in frame_idxs or t == 0:
                
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])
                normalized = normalized.cpu().numpy().astype(np.uint8)

                # Convert single channel to RGB by stacking
                if c == 1:
                    normalized = np.concatenate((normalized,)*3, axis=1)  # Concatenate along the channel axis

                sqrt_n = int(np.sqrt(n_samples))
                if sqrt_n ** 2 == n_samples:  # Vérifier si n_samples est un carré parfait
                    frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=sqrt_n)
                else:
                    frame = einops.rearrange(normalized, "b c h w -> (b h) w c")

                # Rendering frame
                frames.append(frame)

    # Storing the gif
    if gif:
        with imageio.get_writer(gif_name, mode="I") as writer:
            for idx, frame in enumerate(frames):
                writer.append_data(frame)
                if idx == len(frames) - 1:
                    for _ in range(frames_per_gif // 3):
                        writer.append_data(frames[-1])

    return x


def training_loop(ddpm, loader, n_epochs, optim, device=device, debug=debug, store_path=model_path):
    """
    Trains a DDPM model on a given dataset using the specified optimizer and number of epochs.
    
    Args:
        ddpm (DDPM): The DDPM model to be trained.
        loader (DataLoader): The data loader containing the training dataset.
        n_epochs (int): The number of epochs to train the model.
        optim (Optimizer): The optimizer to use for training.
        device (str, optional): The device to use for training. Defaults to the global device variable.
        debug (bool, optional): Whether to enable debug mode. Defaults to the global debug variable.
        store_path (str, optional): The path to store the trained model. Defaults to the global model_path variable.
        
    Returns:
        None
        
    Raises:
        None
        
    Side Effects:
        - Trains the DDPM model on the specified dataset.
        - Stores the trained model at the specified store_path if it is the best model encountered so far.
        - Saves the training loss plot at img_path/plot/plot_loss.png if debug mode is enabled.
        - Saves the generated images at img_path/gif/sampling.gif and img_path/epoch/{epoch + 1} if debug mode is enabled.
        - Prints the loss at each epoch and whether the model is the best encountered so far.
    """
    # Defining loss
    mse = nn.MSELoss()
    best_loss = float("inf")
    
    # Defining steps
    n_steps = ddpm.n_steps

    # Create debug path if needed
    if debug:
        losses = []
        make_path(f"{img_path}/gif")
        make_path(f"{img_path}/epoch")
        make_path(f"{img_path}/plot")
    
    # Training
    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            x0 = batch[0].to(device)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Storing loss
            epoch_loss += loss.item() * len(x0) / len(loader.dataset)
            
        # Plot the loss at each epoch
        if debug:
            losses.append(epoch_loss)
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch + 2), losses, 'o-', label='Loss per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss over Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{img_path}/plot/plot_loss.png")
            plt.close()  

            # Save images generated at this epoch
            show_images(generate_new_images(ddpm, device=device, gif_name=f"{img_path}/gif/sampling.gif"), f"epoch/{epoch + 1}")

        # Print losss
        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)

# main
if __name__ == "__main__":
    # Loading the data (converting each image into a tensor and normalizing between [-1, 1])
    load_list = [
        SquarePad(), # Padding
        Resize(img_size), # Resize
        ToTensor(), # To tensor
        Lambda(lambda x: (x - 0.5) * 2)] # Normalization
    
    if channel == 1:
        load_list.append(Grayscale())
    
    transform = Compose(load_list)
    dataset = ImageFolder(root=dataset_path, transform=transform)
    
    # Filtering the dataset if class_name is specified
    if class_name is not None:
        class_index = dataset.class_to_idx.get(class_name)
        if class_index is None:
            raise ValueError(f"Class {class_name} not found in the dataset.")
        indices = [i for i, (_, label) in enumerate(dataset.samples) if label == class_index]
        dataset = Subset(dataset, indices)
        
    # Creating the data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optionally, show the first batch of images
    if debug:
        show_first_batch(loader)
    
    # Defining model
    ddpm = MyDDPM(MyUNet(channel, img_size), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

    # Optionally, show the diffusion (forward) process
    if debug:
        show_forward(ddpm, loader, device)
    
    # Création de l'optimiseur
    optim = Adam(ddpm.parameters(), lr=lr)

    # Training
    training_loop(ddpm, loader, n_epochs, optim)