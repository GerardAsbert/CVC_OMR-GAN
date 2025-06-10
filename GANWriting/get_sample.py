import os
import torch
import torch.nn.functional as F
from torch.amp import GradScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import time
import argparse
from load_data import vocab_size, loadData_sample
from network_tro import ConTranModel
import cv2
#os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "/home/gasbert/miniconda3/envs/CVC-OMRGan/lib/python3.8/site-packages/cv2/qt/plugins"
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "/home/gasbert/miniconda3/envs/CVC-GAN-OMR/plugins"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
scaler = GradScaler()
OOV = True
pretrained_rec = False

EARLY_STOP_EPOCH = None
EVAL_EPOCH = 10
show_iter_num = 1
LABEL_SMOOTH = True
Bi_GRU = True
VISUALIZE_TRAIN = True

BATCH_SIZE = 16
lr_dis = 1e-5
lr_gen = 1e-4
lr_rec = 1e-5
top10_cosine_euclidean = []

def sample_data_loader():
    sample_loader = loadData_sample()
    sample_loader = torch.utils.data.DataLoader(dataset=sample_loader.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return sample_loader


def generate_sample_images(sample_loader, run_id=0):
    print(f"Device: {device}")
    model_folder = "/home/gasbert/Desktop/CVC_OMR-GAN/GANWriting/weights/save_weights_run" + str(run_id)
    for file in os.listdir(model_folder):
        if file.startswith("GAN") and file.endswith("model") and os.path.isfile(os.path.join(model_folder, file)):
            images = []
            model = ConTranModel(show_iter_num, OOV, pretrained_rec=pretrained_rec, pretrained_gan=os.path.join(model_folder, file)).to(device)
            print(file)
            for i, sample_data_list in enumerate(sample_loader):
                tr_img, tr_label = sample_data_list
                tr_img = tr_img.to(device)
                tr_label = tr_label.to(device)
                sample_data_list = (tr_img, tr_label)

                image = model(sample_data_list, 0, 'get_sample')
                if image.shape[1] < 2048:
                    new_shape = (512, 2048)
                    padded_array = np.zeros(new_shape, dtype=image.dtype)
                    padded_array[:, :image.shape[1]] = image
                    image = padded_array

                images.append(image)
            stacked_image = np.vstack(images)
            cv2.imwrite(model_folder + "/" + file[:-6] + "_sampleimg" +'.jpg', stacked_image)

    torch.cuda.empty_cache()



if __name__ == '__main__':
    print(f"Network_tro.py vocab_size: {vocab_size}")
    start_time = time.ctime()
    print(start_time)
    sample_loader = sample_data_loader()
    generate_sample_images(sample_loader)
    print(time.ctime() - start_time)