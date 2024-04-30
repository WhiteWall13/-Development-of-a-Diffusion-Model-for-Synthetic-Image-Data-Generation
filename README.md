# Development of a Diffusion Model for Synthetic Image Data Generation

This project aims to develop a diffusion model capable of generating synthetic images for various applications. The project architecture is designed to be simple yet effective, allowing for easy customization and experimentation through a detailed configuration file.

## Project Structure

```plaintext
.
├── config.ini      # Configuration file to set up the model parameters
├── doc             # Documentation and training reports
│   ├── train.html
│   ├── train.pdf
│   ├── unet.html
│   ├── unet.pdf
│   ├── rapport.pdf
│   ├── video.mp4
│   └── presentation.pdf
├── output.py       # Script to generate images from the trained model
├── train.py        # Script to start the model training
└── unet.py         # Definition of the U-Net model used for diffusion
```

## Documentation

The following documents are included in the `doc` folder to help understand and utilize the model:

- `train.html` and `train.pdf`: Detailed documentation of the training process.
- `unet.html` and `unet.pdf`: Explanations about the architecture of the U-Net model used.
- `rapport_projet.pdf`: Comprehensive project report, including methodologies, results, and conclusions.
- `presentation.pdf`: PDF format presentation of the project for reviews or demonstrations.
- `video.mp4`: Video presentation of the project, showcasing the model's capabilities and use cases.


## Configuration

The `config.ini` file is at the heart of the model's customization. Here are the main configurable parameters:

```ini
[DEFAULT]
dataset = Path to training data
class = Option to filter data by class
model = Path to the pre-trained model
img_size = Image size after resizing for training
n_channel = Number of image channels (1 for grayscale images, 3 for color images, RGB type)
batch_size = Number of images per batch during training
n_epochs = Total number of training cycles
lr = Learning rate for the optimizer
n_steps = Number of steps in the diffusion process
min_beta = Minimum parameter controlling the intensity of the noise
max_beta = Maximum parameter controlling the intensity of the noise
debug = Enable debug mode to visualize images and graphs during training

[Output]
n_images = Number of images generated to visualize the results
output_path = Path where the generated images will be saved
```

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

Modify the `config.ini` file as needed, then to start the training process, launch:

```bash
python train.py
```

After training, to generate images:

```bash
python output.py
```

## Results Showcase

The generated images can be compared to the initial dataset images to assess the quality and fidelity of the results. Below are some visual comparisons that highlight our model's capabilities:

### Dataset Images:
![Image](https://perso.esiee.fr/~hameaun/projetE4/Original_images.png)

### End of Model Training:
![Image](https://perso.esiee.fr/~hameaun/projetE4/200.png)

### Data Denoising:
![Image](https://perso.esiee.fr/~hameaun/projetE4/sampling.gif)

### Side-by-Side Comparison:
<p align="center">
  <img src="https://perso.esiee.fr/~hameaun/projetE4/output_4.jpg" alt="Generated Image" width="49%" style="float: left; margin-right: 2%;"/>
  <img src="https://perso.esiee.fr/~hameaun/projetE4/dataset.png" alt="Dataset Image" width="49%" style="float: right; margin-left: 2%;"/>
</p>
