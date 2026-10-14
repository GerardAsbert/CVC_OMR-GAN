import os
import re
import math
import random
import shutil
import torch
import glob
from torch import optim
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import argparse
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

    # Input dataset (parent folder of the author folders) and the seeded subset
    # to read from it. Keep these equal to main_run.py's DEFAULT_CONFIG so
    # inference only sees training images. (When called from main_run at the
    # end of training, its values are passed in.)
    "data_root": "/data/113-2/users/gasbert/CYPHERS/copiale/images/symbol_level/all_clean",
    "dataset_percentage": 10,
    "dataset_subset_seed": 0,

    # Where the standalone script looks for checkpoints. Training writes its
    # top-15 models to <model_folder>/save_weights_run<run_id>/GAN-<id>-<score>.model,
    # so point this there when running this script by hand.
    "models_folder":    "/data/113-2/users/gasbert/music-symbol-GAN/results/weights",

    # SNN-style new-sample generation from saved latents. Each (author, class)
    # gets ceil(jitter_percentage% x its number of images) jitters, spread over
    # its images (50 -> half of them get one jitter each; 200 -> two each; 0
    # disables jitter). Which images get the remainder is fixed by
    # dataset_subset_seed.
    "jitter_percentage": 50,
    "jitter_noise_std": None,      # jitter noise level; None -> training-time w_noise

    # Every inference image is saved to disk, but only the first N of each class
    # (per category: reconstructions / jitters) are logged to wandb, since a full
    # dataset can mean tens of thousands of images. None -> log all of them.
    "wandb_images_per_class": 5,

    # Output layout mirrors the SNN's inference_outputs/ tree, under a per-run
    # subfolder <out_dir>/<dataset folder name>/pct<P>_seed<S>/ (see
    # load_data.run_tag), which is emptied at the start of each standalone run:
    #   <out_dir>/training_instances/<class>/inst_<n>.png
    #   <out_dir>/new_jitter/<class>/inst_<n>_new_jitter_<j>.png
    #   <out_dir>/latents/<class>/inst_<n>.npy
    "out_dir":          '/data/113-2/users/gasbert/music-symbol-GAN/results/inference_outputs',

    # On-disk mirror of the inference images in the SAME layout as the input
    # dataset: <out_dir>/<author_symbol_subdir>/<author_name>/<symbol_name>/.
    # Each symbol folder holds BOTH the straight generations (inst_<n>.png) and
    # the jitters (inst_<n>_new_jitter_<j>.png) — the same images logged to
    # wandb, minus the style interpolation. Set to "" to root the author folders
    # directly at out_dir instead of under a subfolder.
    "author_symbol_subdir": "by_author",

    "wandb_project":    "[SNN_OFFLINE_JOURNAL_GENERATION]_GAN",
    "wandb_prefix":     "inference",   # keys become "inference/training_instances", etc.
}

# The symbol classes are read from the folders under data_root, and the model
# modules size themselves from them at import time. When imported by main_run
# the classes are already set from its config; standalone, use GEN_CONFIG's.
import load_data
if not load_data.classes_configured():
    load_data.set_data_root(GEN_CONFIG["data_root"])
from load_data import loadData_generate as load_data_func, author_dirs, run_tag, vocab_size, IMG_WIDTH, IMG_HEIGHT, loadData_imp
from network_tro import ConTranModel
from modules_tro import reset_image_buffers, log_image_buffers
from loss_tro import CER

OUT_DIR = os.path.join(GEN_CONFIG["out_dir"],
                       run_tag(GEN_CONFIG["data_root"], GEN_CONFIG["dataset_percentage"],
                               GEN_CONFIG["dataset_subset_seed"]))
AUTHOR_SUBDIR = GEN_CONFIG["author_symbol_subdir"]


def _folders(out_dir):
    """The SNN's inference_outputs/ layout, rooted at any out_dir, plus a fourth
    root that mirrors the images in the input dataset's
    <author_name>/<symbol_name> layout (holding both straight generations and
    jitters). Creates out_dir and all four roots if they don't exist."""
    folders = (os.path.join(out_dir, "training_instances"),
               os.path.join(out_dir, "new_jitter"),
               os.path.join(out_dir, "latents"),
               os.path.join(out_dir, AUTHOR_SUBDIR) if AUTHOR_SUBDIR else out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for f in folders:
        os.makedirs(f, exist_ok=True)
    return folders


FINAL_FOLDER, JITTER_FOLDER, LATENT_FOLDER, AUTHOR_FOLDER = _folders(OUT_DIR)

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


def all_data_loader(bg_color=None, data_root=None, dataset_percentage=None, dataset_subset_seed=None):
    test_loader = load_data_func(
        directories=author_dirs(data_root or GEN_CONFIG["data_root"]),
        bg_color=bg_color or GEN_CONFIG["bg_color"],
        dataset_percentage=(GEN_CONFIG["dataset_percentage"] if dataset_percentage is None
                            else dataset_percentage),
        subset_seed=(GEN_CONFIG["dataset_subset_seed"] if dataset_subset_seed is None
                     else dataset_subset_seed))
    test_loader = torch.utils.data.DataLoader(dataset=test_loader.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader


def jitter_counts_for(dataset, jitter_percentage, seed):
    """How many jitters each sample gets, aligned with dataset.data.

    Each (author, class) gets ceil(jitter_percentage% x its image count) jitters,
    spread as evenly as possible over its images: every image gets the whole
    rounds, and a seeded pick of images gets one more for the remainder."""
    assert jitter_percentage >= 0, f"jitter_percentage must be >= 0, got {jitter_percentage!r}"
    groups = {}
    for idx, (_, label, author) in enumerate(dataset.data):
        groups.setdefault((author, label), []).append(idx)
    counts = [0] * len(dataset.data)
    for (author, label), idxs in groups.items():
        total = math.ceil(len(idxs) * jitter_percentage / 100)
        rounds, extra = divmod(total, len(idxs))
        lucky = set(random.Random(f"{seed}/{author}/{label}").sample(idxs, extra))
        for i in idxs:
            counts[i] = rounds + (i in lucky)
    return counts


def generate_base_images(test_loader, epoch, modelFile_o_model, out_dir=None, jitter_counts=None):
    """`jitter_counts` is jitter_counts_for() of test_loader.dataset; it is
    matched to the batches by position, so the loader must not shuffle."""
    final_folder, jitter_folder, latent_folder, author_folder = _folders(out_dir or OUT_DIR)
    if jitter_counts is not None:
        assert isinstance(test_loader.sampler, torch.utils.data.SequentialSampler), \
            "jitter_counts needs an unshuffled loader"
        assert len(jitter_counts) == len(test_loader.dataset)
    offset = 0
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
        # The generate loader yields (img, label, author); other loaders yield
        # (img, label). Handle both so this stays robust if fed a plain loader.
        if len(test_data_list) == 3:
            tr_img, tr_label, tr_author = test_data_list
        else:
            tr_img, tr_label = test_data_list
            tr_author = None
        tr_img = tr_img.to(device)
        tr_label = tr_label.to(device)
        batch_counts = (jitter_counts[offset:offset + tr_img.size(0)]
                        if jitter_counts is not None else None)
        offset += tr_img.size(0)

        test_data_list = (tr_img, tr_label)

        #l_dis, l_rec = model(test_data_list, epoch, 'eval', cer_te)
        l_dis, l_rec = model(
            test_data_list, epoch, 'eval',
            final_folder=final_folder,
            jitter_counts=batch_counts,
            jitter_noise_std=GEN_CONFIG["jitter_noise_std"],
            latent_folder=latent_folder,
            jitter_folder=jitter_folder,
            authors=tr_author,
            author_folder=author_folder,
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
                  bg_color=None, augment=False, wandb_project=None,
                  data_root=None, dataset_percentage=None, dataset_subset_seed=None,
                  jitter_percentage=None, wandb_images_per_class="config"):
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

    `jitter_percentage` (None -> GEN_CONFIG) sets the per-(author, class) jitter
    count, see jitter_counts_for. `wandb_images_per_class` caps the images of
    each class logged to wandb ("config" -> GEN_CONFIG, None -> no cap).
    """
    out_dir = out_dir or OUT_DIR
    wandb_prefix = GEN_CONFIG["wandb_prefix"] if wandb_prefix is None else wandb_prefix

    # DataParallel wrappers don't survive the eval path (each replica would write
    # its own files), so always run on the underlying module.
    base = model.module if isinstance(model, torch.nn.DataParallel) else model
    base = base.to(device)
    base.eval()

    if test_loader is None:
        test_loader = all_data_loader(bg_color=bg_color, data_root=data_root,
                                      dataset_percentage=dataset_percentage,
                                      dataset_subset_seed=dataset_subset_seed)

    own_run = wandb.run is None
    if own_run:
        wandb.init(project=wandb_project or GEN_CONFIG["wandb_project"],
                   job_type="inference", config=GEN_CONFIG)

    # Clear accumulated wandb images / instance numbering for this pass.
    if wandb_images_per_class == "config":
        wandb_images_per_class = GEN_CONFIG["wandb_images_per_class"]
    reset_image_buffers(max_per_class=wandb_images_per_class)

    jitter_counts = jitter_counts_for(
        test_loader.dataset,
        GEN_CONFIG["jitter_percentage"] if jitter_percentage is None else jitter_percentage,
        GEN_CONFIG["dataset_subset_seed"] if dataset_subset_seed is None else dataset_subset_seed)
    print(f"[inference] generating into {out_dir}: {len(jitter_counts)} images "
          f"+ {sum(jitter_counts)} jitters ...")
    loss_dis, loss_rec = generate_base_images(test_loader, 0, base, out_dir=out_dir,
                                              jitter_counts=jitter_counts)

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
    reset_image_buffers(max_per_class=GEN_CONFIG["wandb_images_per_class"])

    # Start from an empty run folder so a previous run's images don't linger.
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    jitter_counts = jitter_counts_for(test_loader.dataset, GEN_CONFIG["jitter_percentage"],
                                      GEN_CONFIG["dataset_subset_seed"])

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
            test_loader, 0, os.path.join(models_folder, highest_file), out_dir=OUT_DIR,
            jitter_counts=jitter_counts)

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