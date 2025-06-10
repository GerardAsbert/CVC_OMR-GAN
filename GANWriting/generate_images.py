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
from loss_tro import CER
from PIL import Image
#os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "/home/gasbert/miniconda3/envs/CVC-OMRGan/lib/python3.8/site-packages/cv2/qt/plugins"
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = "/home/gasbert/miniconda3/envs/CVC-GAN-OMR/plugins"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('start_epoch', type=int, help='load saved weights from which epoch')
args = parser.parse_args()

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
top15_cosine_euclidean = []
models_to_use = [0, 2, 4, 5, 7, 8]
shear_factor = 0.2
FINAL_FOLDER = '../../../../data/gasbert/imagesGerard_handwritten/finalImages'


CurriculumModelID = args.start_epoch
def all_data_loader():
    test_loader = load_data_func()
    test_loader = torch.utils.data.DataLoader(dataset=test_loader.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return test_loader


def generate_base_images(test_loader, epoch, modelFile_o_model):
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
        l_dis, l_rec = model(test_data_list, epoch, 'eval', final_folder=FINAL_FOLDER)

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
            if filename.lower().endswith('jpg'):
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


def main(test_loader):
    print(f"Device: {device}")
    model = ConTranModel(show_iter_num, OOV, pretrained_rec=pretrained_rec).to(device)
    print(model)

    models_folder = "save_weights"

    for id in models_to_use:
        highest_file = None
        highest_number = float('-inf')  # Start with negative infinity

        # Iterate through all files in the directory
        for file in os.listdir(models_folder):
            if file.startswith("GAN-" + str(id)) and file.endswith(".model"):
                # Extract the number using a regex
                match = re.search(r"GAN-0-(\d+\.\d+)\.model", file)
                if match:
                    number = float(match.group(1))  # Convert to float
                    # Update if this number is higher
                    if number > highest_number:
                        highest_number = number
                        highest_file = file
        print("Highest file: " + str(highest_file))

        l_dis_test, l_rec_test = generate_base_images(test_loader, 0, models_folder + "/" + highest_file)
    
    augment_images(models_folder)
    print("All images generated! :D")
    

        



def rm_old_model(index):
    models = glob.glob('save_weights/*.model')
    for m in models:
        epoch = int(m.split('.')[0].split('-')[1])
        if epoch < index:
            os.system('rm save_weights/contran-' + str(epoch) + '.model')

if __name__ == '__main__':
    print(f"Network_tro.py vocab_size: {vocab_size}")
    print(time.ctime())
    test_loader = all_data_loader()
    main(train_loader, test_loader, imp_loader)
    print(time.ctime())