import os
import torch
import glob
from torch import optim
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from torch.amp import GradScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from torchvision import transforms
from load_data import loadData as load_data_func, vocab_size, IMG_WIDTH, IMG_HEIGHT
from network_tro import ConTranModel
from loss_tro import CER
from collections import Counter

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('start_epoch', type=int, help='load saved weights from which epoch')
args = parser.parse_args()

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
scaler = GradScaler("cuda")
OOV = True

NUM_THREAD = 4

EARLY_STOP_EPOCH = None
EVAL_EPOCH = 10
MODEL_SAVE_EPOCH = 1
show_iter_num = 1
LABEL_SMOOTH = True
Bi_GRU = True
VISUALIZE_TRAIN = True

BATCH_SIZE = 4
lr_dis = 1e-5
lr_gen = 1e-4
lr_rec = 1e-5

CurriculumModelID = args.start_epoch

def all_data_loader():
    train_loader, validation_loader = load_data_func(OOV, pretrain_rec=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_loader.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_loader.dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    balance_data_loader(train_loader)
    return train_loader, validation_loader

def balance_data_loader(loader):
    labels = []
    for _, tr_label in loader:
        labels.extend(tr_label.cpu().numpy())
    counter = Counter(labels)
    print(f"Label distribution: {counter}")


def train(train_loader, model, rec_opt, epoch):
    model.train()
    
    loss_rec = list()
    loss_rec_tr = list()
    batch_accuracies = []
    time_s = time.time()
    cer_tr = CER()
    cer_te = CER()
    cer_te2 = CER()
    ssim_scores = []

    for i, train_data_list in enumerate(train_loader):
        tr_img, tr_label = train_data_list
        tr_img = tr_img.to(device)
        tr_label = tr_label.to(device)

        #debug_train_data((tr_img, tr_label), "Original")

        train_data_list = (tr_img, tr_label)

        
        '''rec pretrain'''
        rec_opt.zero_grad()
        l_rec_tr, accuracy = model(train_data_list, epoch, 'rec_pretrain_train')
        scaler.scale(l_rec_tr).backward(retain_graph=True)
        scaler.step(rec_opt)
        scaler.update()
        loss_rec_tr.append(l_rec_tr.cpu().item())
        batch_accuracies.append(accuracy)

        # Asegúrate de que los datos no tengan valores anómalos
        for data in train_data_list:
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f'Anomalous data detected in batch {i}')

        # Chequear valores de pérdida
        if torch.isnan(l_rec_tr).any() or torch.isinf(l_rec_tr).any():
            print(f'l_rec_tr: Anomalous loss detected in rec_update at batch {i}')
              

        # Limpiar memoria no utilizada
        del l_rec_tr
        torch.cuda.empty_cache()

    fl_rec = np.mean(loss_rec)
    fl_rec_tr = np.mean(loss_rec_tr)
    
    return batch_accuracies


def test(test_loader, model, epoch):
    model.eval()  # Set the model to evaluation mode
    
    loss_rec = list()
    batch_accuracies = []
    cer_te = CER()
    cer_te2 = CER()
    ssim_scores = []

    time_s = time.time()

    with torch.no_grad():  # Disable gradient calculation
        for i, test_data_list in enumerate(test_loader):
            te_img, te_label = test_data_list
            te_img = te_img.to(device)
            te_label = te_label.to(device)

            test_data_list = (te_img, te_label)

            '''rec test'''
            l_rec_te, accuracy  = model(test_data_list, epoch, 'rec_pretrain_test')  # Keep same mode for pretrain eval
            
            #loss_rec.append(l_rec_te.cpu().item())
            batch_accuracies.append(accuracy)

            # Ensure no anomalous values
            for data in test_data_list:
                if torch.isnan(data).any() or torch.isinf(data).any():
                    print(f'Anomalous data detected in batch {i}')

              
    fl_rec = np.mean(loss_rec)
    
    return batch_accuracies


def main(train_loader, validation_loader):
    print(f"Device: {device}")
    model = ConTranModel(show_iter_num, OOV, pretrained_rec=False).to(device)
        
    rec_params = list(model.rec.parameters())
    rec_opt = optim.Adam([p for p in rec_params if p.requires_grad], lr=lr_rec)

    epochs = 50001
    min_cer = 1e5
    min_idx = 0
    min_count = 0
    train_accuracies = []
    test_accuracies = []

    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"execution_{time.strftime('%Y%m%d-%H%M%S')}.txt")

    for epoch in range(CurriculumModelID, epochs):
        print("Epoch: " + str(epoch))
        batch_accuracies_train = train(train_loader, model, rec_opt, epoch)
        train_accuracies.append(np.array(batch_accuracies_train).mean())
        print("Train accuracy: ")
        print(train_accuracies)

        batch_accuracies_test = test(test_loader, model, epoch)
        test_accuracies.append(np.array(batch_accuracies_test).mean())
        print("Test accuracy: ")
        print(test_accuracies)
        

        if epoch % MODEL_SAVE_EPOCH == 0:
            folder_weights = 'save_weights/44_classes'
            if not os.path.exists(folder_weights):
                os.makedirs(folder_weights)
            torch.save(model.rec.state_dict(), folder_weights + '/recognizer-%d-HW_badsharp.model' % epoch)
    
        
    i = 0
    with open(log_file, "a") as f:
        for line in train_accuracies:
            f.write(str(i) + ": " + str(line) + "\n")


if __name__ == '__main__':
    print(f"Network_tro.py vocab_size: {vocab_size}")
    print(time.ctime())
    train_loader, test_loader = all_data_loader()
    main(train_loader, test_loader)
    print(time.ctime())