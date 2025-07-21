# CVC_OMR-GAN
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/GerardAsbert/CVC_OMR-GAN)

## Overview

This repository contains the implementation of a Generative Adversarial Network (GAN) for Optical Music Recognition (OMR). The primary goal is to generate high-quality, synthetic handwritten music symbols to augment training datasets for OMR systems.


## Architecture

The core of the project is the `ConTranModel` which coordinates the training of the three sub-networks.

*   **Generator (`GenModel_FC`)**: Encodes a source image into a feature representation, adds noise, concatenates a one-hot encoded label, and then decodes this combined vector to produce a new image. It can be configured with different encoder and decoder architectures (`base` or `multiple_level`).
*   **Discriminator (`DisModel`)**: A CNN with residual blocks that outputs a prediction score on whether an input image is real or fake. It uses a gradient penalty for training stability.
*   **Identifier (`RecModel_CNN`)**: A CNN-based classifier trained to recognize music symbols. During GAN training, it provides the recognition loss (`l_rec`) to the generator. This forces the generator to create images that are semantically correct and identifiable.

The total loss for the generator is a weighted sum of the discriminator and generator's adversarial loss and the identifier's recognition loss.

## Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/GerardAsbert/CVC_OMR-GAN.git
    cd CVC_OMR-GAN/GANWriting
    ```

2.  Install the required dependencies. It is recommended to use a virtual environment.
    ```sh
    pip install -r requirements.txt
    ```
    This project is built with Python 3.8.18, PyTorch 2.4.1, and cuda-toolkit 12.0.

## Usage

All scripts should be run from the `GANWriting/` directory.

### 1. Training the GAN

To train the full GAN model, run `main_run.py`.

```sh
python main_run.py <start_epoch> --lr_dis <rate> --lr_gen <rate> --lr_id <rate>
```

**Example:**
```sh
python main_run.py 0 --lr_dis 1e-5 --lr_gen 1e-4 --lr_id 1e-5
```
*   `<start_epoch>`: Set to 0 to train from scratch.
*   The script will log metrics to `wandb`. The top 15 models based on a combined metric of cosine similarity, Euclidean distance, and SSIM between real and generated image encodings will be saved in `weights/save_weights_run<run_id>/`.

### 2. Hyperparameter Sweep with `wandb`

The repository is configured for hyperparameter sweeps using `wandb`.

1.  Initialize the sweep using the provided configuration file:
    ```sh
    wandb sweep sweep_config.yaml
    ```
2.  Run the wandb agent with the sweep ID provided by the previous command:
    ```sh
    wandb agent <YOUR_SWEEP_ID>
    ```
    
### 3. Pre-training the Identifier

Before training the full GAN, you can pre-train the identifier network on the real dataset. While this led to worse results, it is still provided as an optional feature.

```sh
python pretrain_recognizer.py 0
```
This will save the trained model weights in `save_weights/`.


### 4. Generating Synthetic Images

After training, you can use the `generate_images.py` script to generate a dataset of synthetic music symbols using the saved models.

```sh
python generate_images.py <start_epoch>
```
*   The `<start_epoch>` argument is required by the script but is not used to select models; the script will automatically find the best-performing models from the `weights/` directory.
*   The generated images will be saved in a specified folder (default is `../../../../data/gasbert/imagesGerard_handwritten/finalImages`). The script also applies augmentations (rotation, shear) to the generated images.

### 5. Visualizing Samples

To visualize the output of your trained models as a basic sample (two images per symbol, to observe quality and variability), you can run `get_sample.py`. This is useful for qualitative evaluation.

```sh
python get_sample.py
```
This will load saved models from `weights/save_weights_run*` and generate sample sheets, saving them in the same directory.

## Project Structure

```
.
└── GANWriting/
    ├── main_run.py           # Main training script for the GAN.
    ├── network_tro.py        # Defines the main ConTranModel containing Gen, Dis, and Rec.
    ├── modules_tro.py        # Defines the architecture of Gen, Dis, and Rec modules.
    ├── loss_tro.py           # Defines loss functions (CER, Label Smoothing, etc.).
    ├── load_data.py          # Data loading and preprocessing for music symbols.
    ├── pretrain_recognizer.py# Script for pre-training the identifier model.
    ├── generate_images.py    # Script to generate final synthetic images.
    ├── get_sample.py         # Script to get sample outputs for visualization.
    ├── run_sweep.py          # Wrapper script for wandb hyperparameter sweeps.
    ├── sweep_config.yaml     # Configuration file for wandb sweeps.
    ├── requirements.txt      # Python dependencies.
    ├── weights/              # Directory where the best model weights are saved.
    └── recognizer/           # Contains code for an alternative seq2seq recognizer.
