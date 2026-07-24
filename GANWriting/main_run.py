import os

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "/miniconda3/envs/music-symbol-gan/lib/python3.8/site-packages/cv2/qt/plugins"
#os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "/home/gasbert/miniconda3/envs/CVC-GAN-OMR/plugins"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import glob
import itertools
from torch import optim
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import numpy as np
import matplotlib

matplotlib.use('Agg')

import wandb
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import trange, tqdm

from load_data import loadData as load_data_func, loadData_full, vocab_size, IMG_WIDTH, IMG_HEIGHT, loadData_imp, index2letter
from network_tro import ConTranModel, set_crit, set_loss_weights
from collections import Counter


device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
scaler = GradScaler()
pretrained_rec = False

CurriculumModelID = 0


# ── central hyperparameter config ─────────────────────────────────────────────
# Everything NOT exposed on argparse lives here, in the same spirit as the SNN's
# DEFAULT_CONFIG, so there is a single source of truth.
DEFAULT_CONFIG = {
    # optimisation / schedule
    "epochs":                  100,
    "batch_size":              92,
    "eval_epoch":              10,
    "show_iter_num":           1,
    "bi_gru":                  True,
    "visualize_train":         True,
    "oov":                     True,

    # GAN loss weights (moved out of network_tro.py's module globals)
    "w_dis":                   1.0,
    "w_rec":                   2.5,
    "w_noise":                 3.0,
    "w_l1":                    0.0,

    # dataset background / padding colour (mirrors the SNN's pad_color)
    "bg_color":                "white",   # "white" -> pad 255, "black" -> pad 0

    # MSE-vs-images-seen learning curve (the fair SNN-vs-GAN artifact)
    "mse_log_every":           20,        # log per-batch reconstruction MSE every N batches

    # image logging — same convention as the SNN (see _log_symbol_images_to_wandb)
    "img_log_every":           100,       # batches between per-symbol image logs (SNN: every 100 iters)
    "train_image_dir":         "train_outputs",   # end-of-training PNGs, like the SNN's output_dir

    # train-until-convergence (same idea/params as the SNN, in epoch units)
    "train_until_convergence": True,     # False -> run `epochs`; True -> stop on MSE plateau
    "convergence_eval_every":  32,         # epochs between full test-set MSE evals
    "convergence_patience":    6,        # #evals with no improvement (> min_delta) before stopping
    "convergence_min_delta":   1e-9,      # min ABSOLUTE MSE drop that counts as an improvement
    "convergence_max_epochs":  15880,      # hard safety cap on epochs in convergence mode
    "convergence_eval_seed":   12345,     # fixed seed so the generator's eval noise is reproducible

    # train and evaluate on the WHOLE dataset (no held-out split). True ->
    # loadData_full; False -> loadData's 90/10 random_split.
    "full_dataset_no_split":   True,
}

# Derive the module-level names the rest of the file uses from the config.
OOV             = DEFAULT_CONFIG["oov"]
EPOCHS          = DEFAULT_CONFIG["epochs"]
BATCH_SIZE      = DEFAULT_CONFIG["batch_size"]
EVAL_EPOCH      = DEFAULT_CONFIG["eval_epoch"]
show_iter_num   = DEFAULT_CONFIG["show_iter_num"]
Bi_GRU          = DEFAULT_CONFIG["bi_gru"]
VISUALIZE_TRAIN = DEFAULT_CONFIG["visualize_train"]
MSE_LOG_EVERY   = DEFAULT_CONFIG["mse_log_every"]
IMG_LOG_EVERY   = DEFAULT_CONFIG["img_log_every"]

images_seen = 0            # cumulative training images seen (the SNN-comparison x-axis)
global_step = 0            # cumulative batches (the SNN's "iter" for image logging)
loss_history = []          # per-logged-batch image MSE, for the end-of-training loss curve
top15_cosine_euclidean = []

def all_data_loader(important_symbols=True):
    bg_color = DEFAULT_CONFIG["bg_color"]
    if DEFAULT_CONFIG["full_dataset_no_split"]:
        train_loader, test_loader = loadData_full(bg_color=bg_color)
    else:
        train_loader, test_loader = load_data_func(bg_color=bg_color)
    train_loader = torch.utils.data.DataLoader(dataset=train_loader.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_loader.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    balance_data_loader(train_loader)

    if important_symbols:
        imp_loader = loadData_imp(bg_color=bg_color)
        imp_loader = torch.utils.data.DataLoader(dataset=imp_loader.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        return train_loader, test_loader, imp_loader
    
    return train_loader, test_loader

def balance_data_loader(loader):
    labels = []
    for _, tr_label in loader:
        labels.extend(tr_label.cpu().numpy())
    counter = Counter(labels)
    print(f"Label distribution: {counter}")

def debug_train_data(train_data_list, stage="Original"):
    tr_img, tr_label = train_data_list
    print(f"{stage} - tr_img: {tr_img.shape}, tr_label: {tr_label}")
    for i in range(len(tr_img)):
        img = tr_img[i].cpu().numpy().squeeze()
        label = tr_label[i].cpu().item()
        print(f"{stage} - Label: {label}, Image min: {img.min()}, max: {img.max()}")
        plt.imshow(img, cmap='gray')
        plt.title(f"{stage} - Label: {label}")
        plt.savefig(f'plots/plot_{i}.png')
        plt.show()

def compute_ssim(img1, img2):
    """Compute SSIM between two images."""
    img1 = img1.detach().squeeze().cpu().numpy()  # Convert to numpy array
    img2 = img2.detach().squeeze().cpu().numpy()  # Convert to numpy array
    if img1.ndim == 2:  # If grayscale, add channel dimension
        img1 = img1[..., np.newaxis]
    if img2.ndim == 2:  # If grayscale, add channel dimension
        img2 = img2[..., np.newaxis]

    return ssim(img1, img2, channel_axis=-1, win_size=7, data_range=img1.max() - img1.min())


def _full_test_mse(model, test_loader, seed):
    """Reconstruction MSE over the WHOLE test set: gen(img, true label) vs the
    ground-truth image, averaged over all pixels. This is the fair learning-curve
    metric (plotted vs images-seen) and the signal the convergence test watches —
    the GAN analogue of the SNN's deterministic full-dataset MSE.

    The generator injects latent noise, so we fix the RNG with `seed` (and restore
    it afterwards) to make the metric a reproducible function of the weights."""
    base = model.module if isinstance(model, nn.DataParallel) else model
    was_training = model.training
    model.eval()

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)

    se_sum, count = 0.0, 0
    with torch.no_grad():
        for tr_img, tr_label in test_loader:
            tr_img = tr_img.to(device)
            tr_label = tr_label.to(device)
            gen_img = base.gen(tr_img, tr_label)
            se_sum += float(((gen_img - tr_img) ** 2).sum().item())
            count += tr_img.numel()

    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    if was_training:
        model.train()
    return se_sum / max(1, count)


def train(train_loader, model, dis_opt, gen_opt, rec_opt, epoch, test_loader, run_id, imp_loader=None):
    global images_seen, global_step
    model.train()

    for i, train_data_list in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Batches", unit="batch")):
        tr_img, tr_label = train_data_list
        tr_img = tr_img.to(device)
        tr_label = tr_label.to(device)

        #debug_train_data((tr_img, tr_label), "Original")
        train_data_list = (tr_img, tr_label)

        if imp_loader != None:
            # Train on difficult symbols
            if i % 150 == 0 and i != 0:
                for index_imp, imp_data_list in enumerate(imp_loader):
                    if index_imp == 50:
                        break
                    imp_img, imp_label = imp_data_list
                    imp_img = imp_img.to(device)
                    imp_label = imp_label.to(device)
                    imp_data_list = (imp_img, imp_label)


                    #rec update
                    if not pretrained_rec:
                        rec_opt.zero_grad()
                        l_rec_tr = model(imp_data_list, epoch, 'rec_update')
                        scaler.scale(l_rec_tr).backward(retain_graph=True)
                        scaler.step(rec_opt)
                        scaler.update()
                        wandb.log({
                            "epoch": epoch,
                            "batch": i + (index_imp+1)/100,
                            "identifier_loss (id_train)": l_rec_tr.cpu().item(),
                        })

                    #dis update
                    dis_opt.zero_grad()
                    l_dis_tr = model(imp_data_list, epoch, 'dis_update')
                    scaler.scale(l_dis_tr).backward(retain_graph=True)
                    scaler.step(dis_opt)
                    scaler.update()

                    #gen update
                    gen_opt.zero_grad()
                    l_total, l_dis, l_rec, mean_cosine_sim, mean_euclidean, mean_ssim = model(imp_data_list, epoch, 'gen_update')
                    scaler.scale(l_total).backward(retain_graph=True)
                    scaler.step(gen_opt)
                    scaler.update()

                    wandb.log({
                        "epoch": epoch,
                        "batch": i + (index_imp+1)/100,
                        "discriminator_loss (dis_train)": l_dis.cpu().item(),
                        "identifier_loss (gen_train)": l_rec.cpu().item(),
                        "discriminator_loss (gen_train)": l_dis_tr.cpu().item(),
                        "total_loss": l_total.cpu().item()
                    })


                    del l_total, l_dis, l_rec, l_dis_tr
                    torch.cuda.empty_cache()

        #rec update
        if not pretrained_rec:
            rec_opt.zero_grad()
            l_rec_tr = model(train_data_list, epoch, 'rec_update')

            # reduce across GPUs if using DataParallel
            if isinstance(model, nn.DataParallel):
                if l_rec_tr.dim() > 0:
                    l_rec_tr = l_rec_tr.mean()

            scaler.scale(l_rec_tr).backward(retain_graph=True)
            scaler.step(rec_opt)
            scaler.update()
            wandb.log({
                "epoch": epoch,
                "batch": i,
                "identifier_loss (id_train)": l_rec_tr.cpu().item(),
            })

        
        '''dis update'''
        dis_opt.zero_grad()
        l_dis_tr = model(
            train_data_list, epoch, 'dis_update'
        )

        # reduce across GPUs if using DataParallel
        if isinstance(model, nn.DataParallel):
            if l_dis_tr.dim() > 0:
                l_dis_tr = l_dis_tr.mean()

        scaler.scale(l_dis_tr).backward(retain_graph=True)
        scaler.step(dis_opt)
        scaler.update()

        
        '''gen update'''
        gen_opt.zero_grad()
        (
            l_total,
            l_dis,
            l_rec,
            mean_cosine_sim,
            mean_euclidean,
            mean_ssim,
            pred_classes,
            gt_labels
        ) = model(train_data_list, epoch, 'gen_update')

        # reduce across GPUs if using DataParallel
        if isinstance(model, nn.DataParallel):
            if l_rec.dim() > 0:
                l_rec = l_rec.mean()
            if l_dis.dim() > 0:
                l_dis = l_dis.mean()
            if l_total.dim() > 0:
                l_total = l_total.mean()
        scaler.scale(l_total).backward(retain_graph=True)
        scaler.step(gen_opt)
        scaler.update()

        wandb.log({
            "epoch": epoch,
            "batch": i,
            "discriminator_loss (gen_train)": l_dis.cpu().item(),
            "identifier_loss (gen_train)": l_rec.cpu().item(),
            "discriminator_loss (dis_train)": l_dis_tr.cpu().item(),
            "total_loss": l_total.cpu().item()
        })

        # ── MSE vs images-seen: the fair learning-curve axis shared with the SNN.
        #    Count images from the main training path; log a per-batch
        #    reconstruction MSE every MSE_LOG_EVERY batches to bound overhead.
        images_seen += tr_img.size(0)
        global_step += 1
        if (i % MSE_LOG_EVERY) == 0:
            with torch.no_grad():
                base_model = model.module if isinstance(model, nn.DataParallel) else model
                recon = base_model.gen(tr_img, tr_label)
                batch_mse = float(((recon - tr_img) ** 2).mean().item())
            loss_history.append(batch_mse)
            wandb.log({"curve/train_mse": batch_mse, "images_seen": images_seen})

        '''if i in (0, 20):
            wandb.log({
                f"predictions/epoch_{epoch}/batch_{i}_pred_vs_gt": wandb.Table(
                    columns=["index", "predicted", "ground_truth"],
                    data=[
                        [idx, int(p.item()), int(g.item())]
                        for idx, (p, g) in enumerate(
                            zip(pred_classes[:20], gt_labels[:20])
                        )
                    ]
                )
            })'''

        # Per-symbol target-vs-generated figures, same cadence and keys as the SNN
        # (the SNN logs every 100 training iterations).
        if (global_step % IMG_LOG_EVERY) == 0:
            _log_symbol_images_to_wandb(model, tr_img, tr_label, global_step)

        # Asegúrate de que los datos no tengan valores anómalos
        for data in train_data_list:
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f'Anomalous data detected in batch {i}')

        # Chequear valores de pérdida
        if torch.isnan(l_dis_tr).any() or torch.isinf(l_dis_tr).any():
            print(f'ls_dis_tr: Anomalous loss detected in dis_update at batch {i}')
        if torch.isnan(l_total).any() or torch.isinf(l_total).any():
            print(f'ls_total: Anomalous loss detected in gen_update at batch {i}')

        '''# Calcular SSIM
        with torch.no_grad():
            # Generar las imágenes
            generated_imgs = model.gen(tr_img, tr_label)

            # Calcular SSIM para cada par de imágenes
            for gen_img, ref_img in zip(generated_imgs, tr_img):
                ssim_score = compute_ssim(gen_img, ref_img)
                ssim_scores.append(ssim_score)'''
        
        with torch.no_grad():

            if mean_cosine_sim != None:
                model_folder = "/data/113-2/users/gasbert/music-symbol-GAN/results/weights"
                if not os.path.exists(model_folder):
                    os.makedirs(model_folder)
                folder_weights = os.path.join(model_folder, 'save_weights_run' + str(run_id))
                if not os.path.exists(folder_weights):
                    os.makedirs(folder_weights)
                
                cosine_euclidean = (mean_cosine_sim + (1 / (1 + mean_euclidean)) + mean_ssim)/3

                if(len(top15_cosine_euclidean) < 15):
                    id = len(top15_cosine_euclidean)
                    top15_cosine_euclidean.append(cosine_euclidean)                  
                    
                    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    torch.save(state_dict, folder_weights + '/GAN-' + str(id) + '-' + str(cosine_euclidean)[:4] + '.model')
                else:
                    min_value = min(top15_cosine_euclidean)
                    min_index = top15_cosine_euclidean.index(min_value)
                    if(cosine_euclidean > min_value):
                        
                        # Delete worst model
                        prefix = 'GAN-' + str(min_index) + "-" + str(min_value)[:4]
                        for filename in os.listdir(folder_weights):
                            if filename.startswith(prefix):
                                os.remove(os.path.join(folder_weights, filename))
                                break
                        
                        # Save new model
                        top15_cosine_euclidean[min_index] = cosine_euclidean
                        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                        torch.save(state_dict, folder_weights + '/GAN-' + str(min_index) + '-' + str(cosine_euclidean)[:4] + '.model')


        # Limpiar memoria no utilizada
        del l_total, l_dis, l_rec, l_dis_tr
        torch.cuda.empty_cache()


def main(train_loader, test_loader, lr_dis, lr_gen, lr_rec, imp_loader=None, run_id=0, loss_type="CE", multi_gpu=True):

    set_crit(loss_type)
    # Drive the GAN loss weights from the central config. Must run BEFORE the
    # model is built, since w_noise is read in GenModel_FC's constructor.
    set_loss_weights(DEFAULT_CONFIG["w_dis"], DEFAULT_CONFIG["w_rec"],
                     DEFAULT_CONFIG["w_noise"], DEFAULT_CONFIG["w_l1"])

    run_config = {
        **DEFAULT_CONFIG,
        "learning_rate_dis": lr_dis,
        "learning_rate_gen": lr_gen,
        "learning_rate_rec": lr_rec,
        "architecture": "GAN",
        "run_id": run_id,
        "loss_type": loss_type,
    }
    wandb.init(project="[TESTS]snn-handwriting-off-raster", config=run_config)

    # Log the learning curve against images-seen (not the wandb step), so it is
    # directly comparable to the SNN's curve.
    wandb.define_metric("images_seen")
    wandb.define_metric("curve/train_mse", step_metric="images_seen")
    wandb.define_metric("curve/eval_mse",  step_metric="images_seen")

    print(f"Device: {device}")
    model = ConTranModel(show_iter_num, OOV, pretrained_rec=pretrained_rec)

    # Multi-GPU support
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    base_model = model.module if isinstance(model, nn.DataParallel) else model

    if CurriculumModelID > 0:
        model_file = 'save_weights/contran-' + str(CurriculumModelID) + '.model'
        print('Loading ' + model_file)
        model.load_state_dict(torch.load(model_file))
    
    dis_params = list(base_model.dis.parameters())
    gen_params = list(base_model.gen.parameters())
    rec_params = list(base_model.rec.parameters())
    dis_opt = optim.Adam([p for p in dis_params if p.requires_grad], lr=lr_dis)
    gen_opt = optim.Adam([p for p in gen_params if p.requires_grad], lr=lr_gen)
    rec_opt = optim.Adam([p for p in rec_params if p.requires_grad], lr=lr_rec)

    # ── epoch schedule: run a fixed number of epochs, or keep going until the
    #    test MSE plateaus (capped by convergence_max_epochs), mirroring the SNN.
    train_until_convergence = DEFAULT_CONFIG["train_until_convergence"]
    conv_eval_every = DEFAULT_CONFIG["convergence_eval_every"]
    conv_patience   = DEFAULT_CONFIG["convergence_patience"]
    conv_min_delta  = DEFAULT_CONFIG["convergence_min_delta"]
    conv_max_epochs = DEFAULT_CONFIG["convergence_max_epochs"]
    conv_seed       = DEFAULT_CONFIG["convergence_eval_seed"]

    best_mse, n_bad_evals = float("inf"), 0

    if train_until_convergence:
        epoch_iter = itertools.count(CurriculumModelID)
        print(f"[convergence] training until test-MSE plateaus: patience={conv_patience} "
              f"evals @ every {conv_eval_every} epoch(s), min_delta={conv_min_delta}, "
              f"max_epochs={conv_max_epochs}")
    else:
        epoch_iter = trange(CurriculumModelID, EPOCHS, desc="Training Epochs", unit="epoch")

    for epoch in epoch_iter:
        if train_until_convergence and epoch >= conv_max_epochs:
            print(f"[convergence] hit max_epochs={conv_max_epochs}; stopping.")
            break

        print("Epoch: " + str(epoch))
        train(train_loader, model, dis_opt, gen_opt, rec_opt, epoch, test_loader=test_loader, run_id=run_id, imp_loader=imp_loader)

        # Deterministic full test-set reconstruction MSE — the fair learning-curve
        # point and the convergence signal. Logged in BOTH modes so the curve
        # always exists; only used for early stopping when enabled.
        if ((epoch + 1) % conv_eval_every) == 0:
            eval_mse = _full_test_mse(model, test_loader, seed=conv_seed)
            wandb.log({"curve/eval_mse": eval_mse, "images_seen": images_seen})
            print(f"  [eval] epoch {epoch}  test recon MSE = {eval_mse:.6f}")

            if train_until_convergence:
                if eval_mse < best_mse - conv_min_delta:
                    best_mse, n_bad_evals = eval_mse, 0
                else:
                    n_bad_evals += 1
                    print(f"  [convergence] no improvement ({n_bad_evals}/{conv_patience})  "
                          f"eval_mse={eval_mse:.6f}  best={best_mse:.6f}")
                    if n_bad_evals >= conv_patience:
                        print(f"[convergence] test MSE plateaued for {conv_patience} evals — "
                              f"stopping at epoch {epoch}, images_seen={images_seen}, "
                              f"best_mse={best_mse:.6f}.")
                        wandb.log({"convergence/stopped_epoch": epoch,
                                   "convergence/stopped_images_seen": images_seen,
                                   "convergence/best_mse": best_mse})
                        break

        '''if epoch % MODEL_SAVE_EPOCH == 0:
            folder_weights = 'save_weights'
            if not os.path.exists(folder_weights):
                os.makedirs(folder_weights)
            torch.save(model.state_dict(), folder_weights + '/contran-%d.model' % epoch)'''

    # End-of-training analysis (loss curve, val/img_mse, per-symbol PNGs), the
    # same closing step the SNN performs.
    analyze_and_plot(model, test_loader, DEFAULT_CONFIG["train_image_dir"],
                     loss_history=loss_history)

    # ── inference on the freshly-trained model, in the SAME wandb run ─────────
    # The run is still open here, so run_inference logs everything under
    # "inference/…" in this run, exactly as the SNN's train.py does.
    # The import is deliberately local: generate_images sets CUDA_LAUNCH_BLOCKING
    # at import time, which would cripple training speed if imported up top.
    try:
        from generate_images import run_inference
        run_inference(
            model,
            out_dir=os.path.join(DEFAULT_CONFIG["train_image_dir"], "inference"),
            wandb_prefix="inference",
            bg_color=DEFAULT_CONFIG["bg_color"],
        )
    except Exception as e:
        print(f"[inference] skipped due to error: {e}")



def rm_old_model(index):
    models = glob.glob('save_weights/*.model')
    for m in models:
        epoch = int(m.split('.')[0].split('-')[1])
        if epoch < index:
            os.system('rm save_weights/contran-' + str(epoch) + '.model')


# ── image logging, identical in form to the SNN's train.py ────────────────────

def _imshow_any(ax, im):
    if im.ndim == 3:                       # (H, W, 3) colour
        ax.imshow(np.clip(im, 0.0, 1.0))
    else:                                  # (H, W) grayscale
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")


def _imsave_any(path, im):
    if im.ndim == 3:
        plt.imsave(path, np.clip(im, 0.0, 1.0))
    else:
        plt.imsave(path, im, cmap="gray", vmin=0, vmax=1)


def _log_symbol_images_to_wandb(model, tr_img, tr_label, step):
    """One target-vs-generated figure per symbol, logged under a STABLE key
    'images/<symbol>' so wandb shows each symbol's evolution over training —
    the same convention (and figure layout, titles and cadence) as the SNN's
    _log_character_images_to_wandb.

    The SNN picks the last sequence of each unique symbol; here we pick the last
    occurrence of each class present in the batch."""
    base = model.module if isinstance(model, nn.DataParallel) else model
    labels = tr_label.detach().cpu().numpy().reshape(-1)

    with torch.no_grad():
        gen_batch = base.gen(tr_img, tr_label)   # reconstruction with the TRUE label

    # Collect every class present in the batch and emit ONE wandb.log, so all
    # symbols share a step instead of each one advancing the step counter.
    payload = {}
    for li in sorted(set(int(l) for l in labels)):
        nombre_base = index2letter.get(li, f"class_{li}")
        last_i = np.where(labels == li)[0][-1]

        gen = gen_batch[last_i].detach().cpu().squeeze().numpy()
        tgt = tr_img[last_i].detach().cpu().squeeze().numpy()
        mse = float(np.mean((gen - tgt) ** 2))

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        _imshow_any(axes[0], tgt); axes[0].set_title("target")
        _imshow_any(axes[1], gen); axes[1].set_title("generated")
        fig.suptitle(f"{nombre_base}  iter {step}  MSE:{mse:.4f}"); plt.tight_layout()
        payload[f"images/{nombre_base}"] = wandb.Image(fig)
        plt.close(fig)

    if payload:
        wandb.log(payload)


def analyze_and_plot(model, test_loader, output_dir, loss_history=None):
    """End-of-training analysis, mirroring the SNN's analyze_and_plot: the loss
    curve as 'charts/loss_curve', the overall image MSE as 'val/img_mse', and one
    target-vs-generated PNG per symbol (plus the bare generated image) on disk."""
    os.makedirs(output_dir, exist_ok=True)

    if loss_history:
        fig_loss = plt.figure()
        plt.plot(range(1, len(loss_history) + 1), loss_history)
        plt.xlabel("logged training batch"); plt.ylabel("image MSE loss")
        plt.title("Training loss"); plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_training.png", dpi=300)
        wandb.log({"charts/loss_curve": wandb.Image(fig_loss)})
        plt.close()

    mse_all = _full_test_mse(model, test_loader, seed=DEFAULT_CONFIG["convergence_eval_seed"])
    print(f"Image MSE (all): {mse_all:.6f}")
    wandb.log({"val/img_mse": mse_all})

    # last occurrence of each symbol in the test set, as the SNN does
    base = model.module if isinstance(model, nn.DataParallel) else model
    was_training = model.training
    model.eval()
    best = {}
    with torch.no_grad():
        for tr_img, tr_label in test_loader:
            tr_img = tr_img.to(device)
            tr_label = tr_label.to(device)
            gen_batch = base.gen(tr_img, tr_label)
            labels = tr_label.detach().cpu().numpy().reshape(-1)
            for i, li in enumerate(labels):
                best[int(li)] = (tr_img[i].detach().cpu().squeeze().numpy(),
                                 gen_batch[i].detach().cpu().squeeze().numpy())
    if was_training:
        model.train()

    for li in sorted(best):
        nombre_base = index2letter[li]
        t, g = best[li]
        mse = float(np.mean((g - t) ** 2))

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        _imshow_any(axes[0], t); axes[0].set_title("target")
        _imshow_any(axes[1], g); axes[1].set_title("generated")
        fig.suptitle(f"{nombre_base}  MSE:{mse:.4f}"); plt.tight_layout()
        plt.savefig(f"{output_dir}/{nombre_base}.png", dpi=300); plt.close()
        _imsave_any(f"{output_dir}/{nombre_base}_gen.png", g)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('start_epoch', type=int, help='load saved weights from which epoch')
    parser.add_argument('--lr_dis', type=int, default=1e-5, help='Discriminator learning rate')
    parser.add_argument('--lr_gen', type=int, default=1e-4, help='Generator learning rate')
    parser.add_argument('--lr_id', type=int, default=1e-5, help='Identifier learning rate')
    parser.add_argument('--early_stop_epoch', type=int, default=None, help='Early stop epoch, if None, no early stopping')
    parser.add_argument('--train_important_symbols', action='store_true', help="Do some iterations on specific difficult symbols")
    parser.add_argument('--pretrained_recognizer', action='store_true', help="Use pretrained CNN as the recognizer")
    parser.add_argument('--multi_gpu', action='store_true', help='Use multi-GPU training')
    args = parser.parse_args()

    lr_dis = args.lr_dis
    lr_gen = args.lr_gen
    lr_rec = args.lr_id
    CurriculumModelID = args.start_epoch
    pretrained_rec = args.pretrained_recognizer

    print(f"Network_tro.py vocab_size: {vocab_size}")
    print(time.ctime())
    train_loader, test_loader = all_data_loader(important_symbols=args.train_important_symbols)
    main(train_loader, test_loader, lr_dis=lr_dis, lr_gen=lr_gen, lr_rec=lr_rec, multi_gpu=args.multi_gpu)
    print(time.ctime())