import os
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
from load_data import loadData as load_data_func, vocab_size, IMG_WIDTH, IMG_HEIGHT, loadData_imp
from network_tro import ConTranModel
from loss_tro import CER
from collections import Counter
from statistics import mean
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


CurriculumModelID = args.start_epoch
def all_data_loader():
    train_loader, test_loader = load_data_func()
    imp_loader = loadData_imp()
    train_loader = torch.utils.data.DataLoader(dataset=train_loader.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_loader.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    imp_loader = torch.utils.data.DataLoader(dataset=imp_loader.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    balance_data_loader(train_loader)
    return train_loader, test_loader, imp_loader

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

def train(train_loader, model, dis_opt, gen_opt, rec_opt, epoch, test_loader, imp_loader):
    model.train()
    loss_dis = list()
    loss_dis_tr = list()
    loss_rec = list()

    for i, train_data_list in enumerate(train_loader):
        tr_img, tr_label = train_data_list
        tr_img = tr_img.to(device)
        tr_label = tr_label.to(device)

        #debug_train_data((tr_img, tr_label), "Original")
        train_data_list = (tr_img, tr_label)

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

                del l_total, l_dis, l_rec, l_dis_tr
                torch.cuda.empty_cache()

        #rec update
        if not pretrained_rec:
            rec_opt.zero_grad()
            l_rec_tr = model(train_data_list, epoch, 'rec_update')
            scaler.scale(l_rec_tr).backward(retain_graph=True)
            scaler.step(rec_opt)
            scaler.update()

        
        '''dis update'''
        dis_opt.zero_grad()
        l_dis_tr = model(train_data_list, epoch, 'dis_update')
        scaler.scale(l_dis_tr).backward(retain_graph=True)
        scaler.step(dis_opt)
        scaler.update()

        
        '''gen update'''
        gen_opt.zero_grad()
        l_total, l_dis, l_rec, mean_cosine_sim, mean_euclidean, mean_ssim = model(train_data_list, epoch, 'gen_update')
        scaler.scale(l_total).backward(retain_graph=True)
        scaler.step(gen_opt)
        scaler.update()
                

        loss_dis.append(l_dis.cpu().item())
        loss_dis_tr.append(l_dis_tr.cpu().item())
        loss_rec.append(l_rec.cpu().item())
        

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
                folder_weights = 'save_weights'
                if not os.path.exists(folder_weights):
                    os.makedirs(folder_weights)
                
                cosine_euclidean = (mean_cosine_sim + (1 / (1 + mean_euclidean)) + mean_ssim)/3

                if(len(top15_cosine_euclidean) < 15):
                    id = len(top15_cosine_euclidean)
                    top15_cosine_euclidean.append(cosine_euclidean)                  
                    
                    torch.save(model.state_dict(), folder_weights + '/GAN-' + str(id) + '-' + str(cosine_euclidean)[:4] + '.model')
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
                        torch.save(model.state_dict(), folder_weights + '/GAN-' + str(min_index) + '-' + str(cosine_euclidean)[:4] + '.model')


        # Limpiar memoria no utilizada
        del l_total, l_dis, l_rec, l_dis_tr
        torch.cuda.empty_cache()

    return loss_dis, loss_rec


def main(train_loader, test_loader, imp_loader):
    print(f"Device: {device}")
    model = ConTranModel(show_iter_num, OOV, pretrained_rec=pretrained_rec).to(device)
    print(model)

    if CurriculumModelID > 0:
        model_file = 'save_weights/contran-' + str(CurriculumModelID) + '.model'
        print('Loading ' + model_file)
        model.load_state_dict(torch.load(model_file))
    
    dis_params = list(model.dis.parameters())
    gen_params = list(model.gen.parameters())
    rec_params = list(model.rec.parameters())
    dis_opt = optim.Adam([p for p in dis_params if p.requires_grad], lr=lr_dis)
    gen_opt = optim.Adam([p for p in gen_params if p.requires_grad], lr=lr_gen)
    rec_opt = optim.Adam([p for p in rec_params if p.requires_grad], lr=lr_rec)
    epochs = 50001
    
    for epoch in range(CurriculumModelID, epochs):
        print("Epoch: " + str(epoch))
        loss_dis, loss_rec = train(train_loader, model, dis_opt, gen_opt, rec_opt, epoch, test_loader=test_loader, imp_loader=imp_loader)

        bundle_size = 50
        new_loss_diss = [mean(loss_dis[i:i+bundle_size]) for i in range(0, len(loss_dis), bundle_size)]
        new_loss_rec = [mean(loss_rec[i:i+bundle_size]) for i in range(0, len(loss_rec), bundle_size)]

        #Save loss plots
        plt.figure(figsize=(8,8))
        plt.plot(new_loss_diss, '-b', label='Disc_loss')
        plt.plot(new_loss_rec, '-g', label='Cla_loss')
        plt.xlabel("n batch")
        plt.legend(loc='upper left')
        plt.title("Epoch: " + str(epoch) + " - Losses")
        plt.savefig("loss_plots/Epoch" + str(epoch)+ "Losses.png")

        '''if epoch % MODEL_SAVE_EPOCH == 0:
            folder_weights = 'save_weights'
            if not os.path.exists(folder_weights):
                os.makedirs(folder_weights)
            torch.save(model.state_dict(), folder_weights + '/contran-%d.model' % epoch)'''



def rm_old_model(index):
    models = glob.glob('save_weights/*.model')
    for m in models:
        epoch = int(m.split('.')[0].split('-')[1])
        if epoch < index:
            os.system('rm save_weights/contran-' + str(epoch) + '.model')

if __name__ == '__main__':
    print(f"Network_tro.py vocab_size: {vocab_size}")
    print(time.ctime())
    train_loader, test_loader, imp_loader = all_data_loader()
    main(train_loader, test_loader, imp_loader)
    print(time.ctime())