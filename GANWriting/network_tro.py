import torch
import torch.nn.functional as F
from torch import nn
from load_data import loadData as load_data_func, vocab_size, IMG_WIDTH, IMG_HEIGHT
from modules_tro import GenModel_FC, DisModel, RecModel_CNN as RecModel, write_image, augmentor, randomize_labels, write_image_cosine, return_image, write_final_images
from loss_tro import crit_CE, crit_KL, log_softmax
import numpy as np
import cv2
import random
from skimage.metrics import structural_similarity as ssim


w_dis = 1.
w_l1 = 0.
w_rec = 2.5
w_noise = 3.
#w_fid = 0.001
epsilon = 1e-6
augmentorBool = False
encoderType = 'base'
decoderType = 'base'
crit = crit_CE

def set_crit(loss_type):
    global crit
    if loss_type == "KL":
        crit = crit_KL
    else:
        crit = crit_CE

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
ssim_metric = ssim

class ConTranModel(nn.Module):
    def __init__(self, show_iter_num, oov, pretrained_rec, pretrained_gan=None):
        super(ConTranModel, self).__init__()
        self.gen = GenModel_FC(w_noise = w_noise, encoder_type=encoderType, decoder_type=decoderType).to(device)
        self.dis = DisModel().to(device)
        self.rec = RecModel(vocab_size=vocab_size, pretrained=pretrained_rec).to(device)
        self.iter_num = 0
        self.show_iter_num = show_iter_num
        self.oov = oov
        if pretrained_gan is not None:
            self.load_state_dict(torch.load(pretrained_gan))
            self.eval()

    def forward(self, train_data_list, epoch, mode, cer_func=None, final_folder=None):
        tr_img, tr_label = train_data_list
        noisy_img = tr_img.clone()
        batch_size = tr_img.shape[0]
        mean_cosine_sim = None
        mean_euclidean = None
        mean_ssim = None
        
        if(augmentorBool == True and random.randint(1,5) != 1):
            tr_img = tr_img.to(device)
            for i in range(batch_size):
                subimg = (noisy_img.cpu()[i][0].numpy())*255
                subimg = subimg.astype(np.uint8)
                #cv2.imshow('Original', subimg)
                #cv2.waitKey(0)
                arr = cv2.resize(augmentor(subimg), (subimg.shape[1], subimg.shape[0]), interpolation=cv2.INTER_AREA)
                #cv2.imshow('Noise', arr)
                #cv2.waitKey(0)
                arr = arr.astype(np.float32)
                arr = arr/255
                noisy_img[i][0] = torch.from_numpy(arr)
        #print(f"tr_img: {tr_img}")
        tr_img = tr_img.to(device)
        #print(f"tr_label: {tr_label}")
        tr_label = tr_label.to(device)
        noisy_img = noisy_img.to(device)

        if tr_label.dim() == 1:
            tr_label = tr_label.unsqueeze(1)
        
        if mode == 'rec_update':
            tr_img_rec = tr_img
            tr_img_rec = tr_img_rec.requires_grad_()
            pred_xt_tr = self.rec(tr_img_rec, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH]*batch_size)))
            l_rec_tr = crit(log_softmax(pred_xt_tr.reshape(-1,vocab_size)), tr_label.reshape(-1))
            l_rec_tr.backward(retain_graph=True)
            return l_rec_tr
        
        elif mode == 'gen_update':
            self.iter_num += 1

            tr_label_random = tr_label.clone()
            tr_label_random = randomize_labels(tr_label_random, batch_size, vocab_size)
            generated_img = self.gen(noisy_img, tr_label_random)
            l_dis = self.dis.calc_gen_loss(generated_img)

            pred_xt = self.rec(generated_img, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)).to(device))
            
            '''print("Pred_XT: ")
            print(pred_xt)
            print("Pred_XT_resized: ")
            print(log_softmax(pred_xt.reshape(-1, vocab_size)))
            print("tr_label resized: ")
            print(tr_label.reshape(-1))'''

            log_softmax_pred_xt = log_softmax(pred_xt.reshape(-1, vocab_size))
            l_rec = crit(log_softmax_pred_xt, tr_label.reshape(-1))

            
            l_total = w_dis * l_dis + w_rec * l_rec
            l_total.backward(retain_graph=True)

            '''# Add gradient clipping 
            torch.nn.utils.clip_grad_norm_(self.rec.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.dis.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=1.0)'''
            
            with torch.no_grad():

                if self.iter_num % 10 == 0:
                    # Veure distancia entre els feature vectors de les images gt i les generades
                    tr_encoding = self.gen(noisy_img, tr_label, return_encoding=True)
                    gen_encoding = self.gen(generated_img, tr_label, return_encoding=True)

                    cosine_sim = F.cosine_similarity(tr_encoding.view(batch_size, -1), gen_encoding.view(batch_size, -1), dim=1)
                    mean_cosine_sim = cosine_sim.mean()

                    # Calculate the Euclidean distance
                    euclidean_distance = torch.norm(tr_encoding.view(batch_size, -1) - gen_encoding.view(batch_size, -1), dim=1)
                    mean_euclidean = euclidean_distance.mean()

                    tr_img_nograd = tr_img.detach()
                    mean_cosine_sim = mean_cosine_sim.item()
                    mean_euclidean = mean_euclidean.item()
                    
                    ssim_values= []
                    '''print("Image tensor size: ")
                    print(noisy_img.size())'''
                    generated_img_gpu = generated_img.to(device)
                    for gt_image, synthetic_image in zip(noisy_img, generated_img_gpu):
                        '''print("gt image size:")
                        print(gt_image.size())
                        print("synthetic size:")
                        print(synthetic_image.size())

                        #gt_img = cv2.resize(gt_img, (128, 128))
                        print("Image tensor size after unsqueeze: ")
                        print(gt_image.unsqueeze(0).size())'''
                        min_val = gt_image.min()
                        max_val = gt_image.max()

                        '''print(f"Minimum value in the image: {min_val.item()}")
                        print(f"Maximum value in the image: {max_val.item()}")'''
                        
                        gt_image_np = gt_image.cpu().numpy()
                        synthetic_image_np = synthetic_image.cpu().numpy()
                        gt_image_np = gt_image_np[0, :, :]
                        synthetic_image_np = synthetic_image_np[0, :, :]

                        '''min_val = np.min(synthetic_image_np)
                        max_val = np.max(synthetic_image_np)
                        synthetic_image_np = (synthetic_image_np - min_val) / (max_val - min_val)'''

                        min_val = np.min(synthetic_image_np)
                        max_val = np.max(synthetic_image_np)
                        # TINC QUE NORMALITZAT LES IMATGES GENERADES A (0, 1) ABANS DE COMPUTAR TOTES LES METRIQUES
                        '''print(f"Minimum value in the image: {min_val}")
                        print(f"Maximum value in the image: {max_val}")

                        print("gt image size:")
                        print(gt_image_np.shape)
                        print("synthetic size:")
                        print(synthetic_image_np.shape)'''
                        ssim_value, _ = ssim(gt_image_np, synthetic_image_np, full=True, data_range=1.0)
                        #ssim_value = ssim_metric(gt_image.unsqueeze(0), synthetic_image.unsqueeze(0))
                        ssim_values.append(ssim_value.item())
                    mean_ssim = sum(ssim_values) / len(ssim_values)
                    print("MEAN euclidean: ")
                    print(mean_euclidean)
                    print("MEAN COSINE: ")
                    print(mean_cosine_sim)
                    print("MEAN SSIM: ")
                    print(mean_ssim)
                    print("SSIM VALUES: ")
                    print(ssim_values)
                    write_image_cosine(generated_img, pred_xt, tr_img_nograd, tr_label, self.iter_num, mean_cosine_sim, mean_euclidean, mean_ssim)
                    
                    

                # Convert model predictions to token indices (argmax over vocab dimension)
                pred_classes = torch.argmax(log_softmax_pred_xt, dim=-1)  # Get class predictions (batch_size, timesteps)
                best_softmax = torch.max(log_softmax_pred_xt, dim=-1) 
                tr_label_1dim = tr_label.squeeze(1)
                
                print("Pred classes: ")
                print(pred_classes)
                print("tr label: ")
                print(tr_label_1dim)
                print("---------------------------------")
                
            return l_total, l_dis, l_rec, mean_cosine_sim, mean_euclidean, mean_ssim

        elif mode == 'dis_update':
            sample_img = noisy_img
            sample_img.requires_grad_()
            l_real = self.dis.calc_dis_real_loss(sample_img)
            l_real.backward(retain_graph=True)

            with torch.no_grad():
                generated_img = self.gen(noisy_img, tr_label)

            l_fake = self.dis.calc_dis_fake_loss(generated_img)
            l_fake.backward(retain_graph=True)

            gp = self.dis.gradient_penalty(sample_img, generated_img)
            gp.backward(retain_graph=True)

            l_total = l_real + l_fake
            if self.iter_num % self.show_iter_num == 0:
                with torch.no_grad():
                    pred_xt = self.rec(generated_img, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)).to(device))
                    #pred_xt = self.rec(tr_img)
                    tr_img_nograd = tr_img.detach()
                #write_image(generated_img, pred_xt, tr_img_nograd, tr_label, 'epoch_' + str(epoch) + '-' + str(self.iter_num).zfill(7), self.iter_num)
            return l_total
        
        elif mode == 'rec_pretrain_train':
            self.iter_num += 1
            pred_xt = self.rec(tr_img, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)).to(device))
            #pred_xt = self.rec(tr_img)
            log_softmax_pred_xt = log_softmax(pred_xt.reshape(-1, vocab_size))
            
            l_rec = crit(log_softmax_pred_xt, tr_label.reshape(-1))
            # Calculate accuracy for the batch
            with torch.no_grad():
                pred_classes = torch.argmax(log_softmax_pred_xt, dim=-1)  # Get class predictions (batch_size, timesteps)
                tr_label_1dim = tr_label.squeeze(1)
                
                correct_predictions = (pred_classes == tr_label_1dim).float()  # Compare with ground truth
                accuracy = correct_predictions.mean().item()
                
            l_rec.backward(retain_graph=True)
            
            return l_rec, accuracy
        
        elif mode == 'rec_pretrain_test':
            self.iter_num += 1
            pred_xt = self.rec(tr_img, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)).to(device))
            #pred_xt = self.rec(tr_img)
            log_softmax_pred_xt = log_softmax(pred_xt.reshape(-1, vocab_size))
            
            l_rec = crit(log_softmax_pred_xt, tr_label.reshape(-1))
            
            # Calculate accuracy for the batch
            pred_classes = torch.argmax(log_softmax_pred_xt, dim=-1)  # Get class predictions (batch_size, timesteps)
            tr_label_1dim = tr_label.squeeze(1)
            
            correct_predictions = (pred_classes == tr_label_1dim).float()  # Compare with ground truth
            accuracy = correct_predictions.mean().item()
            
            return l_rec, accuracy

        elif mode == 'eval':
            self.gen.eval()
            self.rec.eval()
            with torch.no_grad():
                generated_img = self.gen(noisy_img, tr_label)
                pred_xt = self.rec(generated_img, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)).to(device))
                #pred_xt = self.rec(tr_img)
                write_final_images(generated_img, pred_xt, tr_img, tr_label, 'eval-' + str(self.iter_num).zfill(7), final_folder)
                self.iter_num += 1
                l_dis = self.dis.calc_gen_loss(generated_img)
                l_rec = crit(log_softmax(pred_xt.reshape(-1, vocab_size)), tr_label.reshape(-1))
                if cer_func:
                    cer_func.add(pred_xt, tr_label)
            return l_dis, l_rec
        
        elif mode == 'get_sample':
            self.gen.eval()
            self.rec.eval()
            with torch.no_grad():
                generated_img = self.gen(noisy_img, tr_label)
                pred_xt = self.rec(generated_img, tr_label, img_width=torch.from_numpy(np.array([IMG_WIDTH] * batch_size)).to(device))
                #pred_xt = self.rec(tr_img)
                image = return_image(generated_img, pred_xt, tr_img, tr_label, 'eval_' + str(epoch) + '-' + str(self.iter_num).zfill(7), self.iter_num)
                self.iter_num += 1
            return image

