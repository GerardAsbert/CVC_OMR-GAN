import os
import re
import torch
import glob
from torch import optim
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from torch.amp import GradScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
from load_data import loadData_generate as load_data_func, vocab_size, IMG_WIDTH, IMG_HEIGHT, loadData_imp
from network_tro import ConTranModel
from modules_tro import reset_image_buffers, log_image_buffers
from loss_tro import CER
from PIL import Image
import wandb
#os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "/home/gasbert/miniconda3/envs/CVC-OMRGan/lib/python3.8/site-packages/cv2/qt/plugins"
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "/home/gasbert/miniconda3/envs/CVC-GAN-OMR/plugins"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
scaler = GradScaler()
# ── central inference config (everything not on argparse), SNN-style ──────────
GEN_CONFIG = {
    "oov":              True,
    "pretrained_rec":   False,
    "eval_epoch":       10,
    "show_iter_num":    1,
    "bi_gru":           True,
    "visualize_train":  True,
    "batch_size":       16,
    "lr_dis":           1e-5,
    "lr_gen":           1e-4,
    "lr_rec":           1e-5,
    "models_to_use":    [0],
    "shear_factor":     0.2,
    "bg_color":         "white",   # MUST match the padding used at training time

    # Where the standalone script looks for checkpoints. Training writes its
    # top-15 models to <model_folder>/save_weights_run<run_id>/GAN-<id>-<score>.model,
    # so point this there when running this script by hand.
    "models_folder":    "/data/113-2/users/gasbert/music-symbol-GAN/results/weights",

    # SNN-style new-sample generation from saved latents
    "n_jitter":         3,         # #new jittered samples per source image (SNN uses 3; 0 disables)
    "jitter_noise_std": None,      # jitter noise level; None -> training-time w_noise

    # Output layout mirrors the SNN's inference_outputs/ tree:
    #   <out_dir>/training_instances/<class>/inst_<n>.png
    #   <out_dir>/new_jitter/<class>/inst_<n>_new_jitter_<j>.png
    #   <out_dir>/latents/<class>/inst_<n>.npy
    "out_dir":          '/data/113-2/users/gasbert/music-symbol-GAN/results/inference_outputs',
    "wandb_project":    "[TESTS]snn-handwriting-off-raster",
    "wandb_prefix":     "inference",   # keys become "inference/training_instances", etc.
}

OUT_DIR = GEN_CONFIG["out_dir"]


def _folders(out_dir):
    """The SNN's inference_outputs/ layout, rooted at any out_dir."""
    return (os.path.join(out_dir, "training_instances"),
            os.path.join(out_dir, "new_jitter"),
            os.path.join(out_dir, "latents"))


FINAL_FOLDER, JITTER_FOLDER, LATENT_FOLDER = _folders(OUT_DIR)

OOV             = GEN_CONFIG["oov"]
pretrained_rec  = GEN_CONFIG["pretrained_rec"]
EVAL_EPOCH      = GEN_CONFIG["eval_epoch"]
show_iter_num   = GEN_CONFIG["show_iter_num"]
Bi_GRU          = GEN_CONFIG["bi_gru"]
VISUALIZE_TRAIN = GEN_CONFIG["visualize_train"]
BATCH_SIZE      = GEN_CONFIG["batch_size"]
lr_dis          = GEN_CONFIG["lr_dis"]
lr_gen          = GEN_CONFIG["lr_gen"]
lr_rec          = GEN_CONFIG["lr_rec"]
models_to_use   = GEN_CONFIG["models_to_use"]
shear_factor    = GEN_CONFIG["shear_factor"]
MODELS_FOLDER   = GEN_CONFIG["models_folder"]
top15_cosine_euclidean = []


def all_data_loader(bg_color=None):
    test_loader = load_data_func(bg_color=bg_color or GEN_CONFIG["bg_color"])
    test_loader = torch.utils.data.DataLoader(dataset=test_loader.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader


def generate_base_images(test_loader, epoch, modelFile_o_model, out_dir=None):
    final_folder, jitter_folder, latent_folder = _folders(out_dir or OUT_DIR)
    if type(modelFile_o_model) == str:
        model = ConTranModel(show_iter_num, OOV, pretrained_rec=False).to(device)
        print('Loading ' + modelFile_o_model)
        model.load_state_dict(torch.load(modelFile_o_model, map_location=device))
    else:
        model = modelFile_o_model
    model.eval()
    loss_dis = list()
    loss_rec = list()
    time_s = time.time()
    cer_te = CER()
    cer_te2 = CER()
    ssim_scores = []
    for test_data_list in test_loader:
        tr_img, tr_label = test_data_list
        tr_img = tr_img.to(device)
        tr_label = tr_label.to(device)

        test_data_list = (tr_img, tr_label)

        #l_dis, l_rec = model(test_data_list, epoch, 'eval', cer_te)
        l_dis, l_rec = model(
            test_data_list, epoch, 'eval',
            final_folder=final_folder,
            n_jitter=GEN_CONFIG["n_jitter"],
            jitter_noise_std=GEN_CONFIG["jitter_noise_std"],
            latent_folder=latent_folder,
            jitter_folder=jitter_folder,
        )

        loss_dis.append(l_dis.cpu().item())
        loss_rec.append(l_rec.cpu().item())


        del l_dis, l_rec
        torch.cuda.empty_cache()

    return loss_dis, loss_rec


def augment_images(input_folder):
    """
    Applies data augmentation (rotation, slant, flip, etc.) to all images in a folder.

    :param input_folder: Path to the folder containing input images.
    :param output_folder: Path to the folder where augmented images will be saved.
    :param num_augmentations: Number of augmented versions per image.
    """

    # Iterate through all files in the input folder
    for symbol in os.listdir(input_folder):
        symbol_folder = os.path.join(input_folder, symbol)
        for filename in os.listdir(symbol_folder):
            index = 1
            # images are now written as .png (SNN convention); .jpg kept for older runs
            if filename.lower().endswith(('.png', '.jpg')):
                # Load the image
                image_path = os.path.join(symbol_folder, filename)
                image = Image.open(image_path).convert('L')
            
                rotated_image = image.rotate(10, expand=True)
                rotated_image.save(image_path[:-4] + '_' + str(index) + image_path[-4:])
                index += 1
            
                rotated_image2 = image.rotate(-10, expand=True)
                rotated_image2.save(image_path[:-4] + '_' + str(index) + image_path[-4:])
                index += 1
                

                shear_matrix_right = (
                    1, 0, 0,  # First row: horizontal shear
                    shear_factor, 1, 0,             # Second row: no vertical shear
                )
                shear_matrix_left = (
                    1, 0, 0,  # First row: horizontal shear
                    -shear_factor, 1, 0,             # Second row: no vertical shear
                )
                width, height = image.size

                # Apply the affine transform with the shear matrix
                new_height = int(height + abs(shear_factor) * width)  # Adjust width based on shear factor
                new_width = width  # Keep the same height

                sheared_image_expanded = image.transform(
                    (new_width, new_height), Image.AFFINE, shear_matrix_right, resample=Image.BICUBIC
                )
                sheared_image_expanded.save(image_path[:-4] + '_' + str(index) + image_path[-4:])
                index += 1

                sheared_image_expanded2 = image.transform(
                    (new_width, new_height), Image.AFFINE, shear_matrix_left, resample=Image.BICUBIC
                )
                sheared_image_expanded2.save(image_path[:-4] + '_' + str(index) + image_path[-4:])
                index += 1

                rotated_image = image.rotate(10, expand=True)
                rotated_image.save(image_path[:-4] + '_' + str(index) + image_path[-4:])
                index += 1
                rotated_image2 = image.rotate(-10, expand=True)
                rotated_image2.save(image_path[:-4] + '_' + str(index) + image_path[-4:])
                index += 1

                rotated_image = sheared_image_expanded.rotate(10, expand=True)
                rotated_image.save(image_path[:-4] + '_' + str(index) + image_path[-4:])
                index += 1
                rotated_image2 = sheared_image_expanded.rotate(-10, expand=True)
                rotated_image2.save(image_path[:-4] + '_' + str(index) + image_path[-4:])
                index += 1

                rotated_image = sheared_image_expanded2.rotate(10, expand=True)
                rotated_image.save(image_path[:-4] + '_' + str(index) + image_path[-4:])
                index += 1
                rotated_image2 = sheared_image_expanded2.rotate(-10, expand=True)
                rotated_image2.save(image_path[:-4] + '_' + str(index) + image_path[-4:])
                index += 1


def run_inference(model, test_loader=None, out_dir=None, wandb_prefix=None,
                  bg_color=None, augment=False, wandb_project=None):
    """Run inference on an ALREADY-TRAINED, in-memory model — the GAN's
    equivalent of the SNN's run_inference, so training can chain straight into
    it without reloading a checkpoint.

    wandb behaviour is identical to the SNN's:
      * A run is already active (called from main_run at the end of training):
        log into THAT run under "<wandb_prefix>/…". Do not init/finish.
      * No active run: start one just for inference and finish it before
        returning.

    `augment` defaults to False because the SNN performs no augmentation at
    inference; the standalone script keeps its original augmentation step.
    """
    out_dir = out_dir or OUT_DIR
    wandb_prefix = GEN_CONFIG["wandb_prefix"] if wandb_prefix is None else wandb_prefix

    # DataParallel wrappers don't survive the eval path (each replica would write
    # its own files), so always run on the underlying module.
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    base = base.to(device)
    base.eval()

    if test_loader is None:
        test_loader = all_data_loader(bg_color=bg_color)

    own_run = wandb.run is None
    if own_run:
        wandb.init(project=wandb_project or GEN_CONFIG["wandb_project"],
                   job_type="inference", config=GEN_CONFIG)

    # Clear accumulated wandb images / instance numbering for this pass.
    reset_image_buffers()

    print(f"[inference] generating into {out_dir} ...")
    loss_dis, loss_rec = generate_base_images(test_loader, 0, base, out_dir=out_dir)

    # One wandb.log per category with the full image list, exactly as the SNN does.
    log_image_buffers(prefix=wandb_prefix)

    if augment:
        augment_images(_folders(out_dir)[0])

    print("[inference] done.")
    if own_run:
        wandb.finish()

    return loss_dis, loss_rec


def main(test_loader):
    print(f"Device: {device}")
    model = ConTranModel(show_iter_num, OOV, pretrained_rec=pretrained_rec).to(device)
    print(model)

    # wandb, handled exactly as the SNN's run_inference: if a run is already
    # active reuse it (so inference lands in the training run under
    # "inference/..."), otherwise own a fresh run and finish it before returning.
    own_run = wandb.run is None
    if own_run:
        wandb.init(project=GEN_CONFIG["wandb_project"], job_type="inference",
                   config=GEN_CONFIG)

    # Clear accumulated wandb images / instance numbering for this pass.
    reset_image_buffers()

    models_folder = MODELS_FOLDER

    for id in models_to_use:
        highest_file = None
        highest_number = float('-inf')  # Start with negative infinity

        # Iterate through all files in the directory
        for file in os.listdir(models_folder):
            if file.startswith("GAN-" + str(id)) and file.endswith(".model"):
                # Extract the score for THIS id (the old pattern hardcoded id 0,
                # so every other id silently found nothing).
                match = re.search(rf"GAN-{id}-(\d+(?:\.\d+)?)\.model", file)
                if match:
                    number = float(match.group(1))  # Convert to float
                    # Update if this number is higher
                    if number > highest_number:
                        highest_number = number
                        highest_file = file
        print("Highest file: " + str(highest_file))

        if highest_file is None:
            print(f"  no checkpoint matching GAN-{id}-*.model in {models_folder}; skipping.")
            continue

        l_dis_test, l_rec_test = generate_base_images(
            test_loader, 0, os.path.join(models_folder, highest_file), out_dir=OUT_DIR)

    # One wandb.log per category with the full image list, exactly as the SNN does:
    # "inference/training_instances" and "inference/new_jitter".
    log_image_buffers(prefix=GEN_CONFIG["wandb_prefix"])

    augment_images(FINAL_FOLDER)
    print("All images generated! :D")

    if own_run:
        wandb.finish()
    

        



def rm_old_model(index):
    models = glob.glob('save_weights/*.model')
    for m in models:
        epoch = int(m.split('.')[0].split('-')[1])
        if epoch < index:
            os.system('rm save_weights/contran-' + str(epoch) + '.model')

if __name__ == '__main__':
    # argparse lives here (not at module scope) so this file can be imported by
    # main_run.py without hijacking that script's command line.
    parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('start_epoch', type=int, help='load saved weights from which epoch')
    args = parser.parse_args()
    CurriculumModelID = args.start_epoch

    print(f"Network_tro.py vocab_size: {vocab_size}")
    print(time.ctime())
    test_loader = all_data_loader()
    main(test_loader)
    print(time.ctime())