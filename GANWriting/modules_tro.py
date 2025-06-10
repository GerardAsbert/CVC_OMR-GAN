import numpy as np
import os
import torch
from torch import nn
from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
from vgg_tro_channel3_modi import vgg19_bn
from recognizer.models.encoder_vgg import Encoder as rec_encoder
from recognizer.models.decoder import Decoder as rec_decoder
from recognizer.models.seq2seq import Seq2Seq as rec_seq2seq
from recognizer.models.attention import locationAttention as rec_attention
from load_data import IMG_HEIGHT, IMG_WIDTH, index2letter, tokens, onehotencoder, TARGET_CLASSES
import cv2
import random

device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')


def augmentor(img):
    TH,TW=img.shape

    param_gamma_low=.3
    #param_gamma_low=.5 # Nacho fixed
    param_gamma_high=2

    param_mean_gaussian_noise=0
    param_sigma_gaussian_noise=100**0.5

    param_kanungo_alpha=2 # params controlling how much foreground and background pixels flip state
    param_kanungo_beta=2
    param_kanungo_alpha0=1
    param_kanungo_beta0=1
    param_kanungo_mu=0
    param_kanungo_k=2

    param_min_shear=-.25 # here a little bit more shear to the left than to the right
    param_max_shear=.25

    param_rotation=2 # plus minus angles for rotation

    param_scale=.2 # one plus minus parameter as scaling factor

    param_movement_BB=6 # translation for cropping errors in pixels

    # add gaussian noise
    gauss = np.random.normal(param_mean_gaussian_noise,param_sigma_gaussian_noise,(TH,TW))
    gauss = gauss.reshape(TH,TW)
    gaussiannoise = np.uint8(np.clip(np.float32(img) + gauss,0,255))

    # randomly erode, dilate or nothing
    # we could move it also after binarization
    kernel=np.ones((2,2),np.uint8)
    #a=random.choice([1,2,3])
    a=random.choice([1,2,3]) # Nacho fixed
    #a = 3 # Nacho fixed
    if a==1:
        gaussiannoise=cv2.dilate(gaussiannoise,kernel,iterations=1)
    elif a==2:
        gaussiannoise=cv2.erode(gaussiannoise,kernel,iterations=1)

    # add random gamma correction
    gamma=np.random.uniform(param_gamma_low,param_gamma_high)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    gammacorrected = cv2.LUT(np.uint8(gaussiannoise), table)

    # binarize image with Otsu
    otsu_th,binarized = cv2.threshold(gammacorrected,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Kanungo noise
    dist = cv2.distanceTransform(1-binarized, cv2.DIST_L1, 3)  # try cv2.DIST_L1 for newer versions of OpenCV
    dist2 = cv2.distanceTransform(binarized, cv2.DIST_L1, 3) # try cv2.DIST_L1 for newer versions of OpenCV

    dist = dist.astype('float64') # Tro add
    dist2 = dist2.astype('float64') # Tro add

    P=(param_kanungo_alpha0*np.exp(-param_kanungo_alpha * dist**2)) + param_kanungo_mu
    P2=(param_kanungo_beta0*np.exp(-param_kanungo_beta * dist2**2)) + param_kanungo_mu
    distorted=binarized.copy()
    distorted[((P>np.random.rand(P.shape[0],P.shape[1])) & (binarized==0))]=1
    distorted[((P2>np.random.rand(P.shape[0],P.shape[1])) & (binarized==1))]=0
    closing = cv2.morphologyEx(distorted, cv2.MORPH_CLOSE, np.ones((param_kanungo_k,param_kanungo_k),dtype=np.uint8))

    # apply binary image as mask and put it on a larger canvas
    pseudo_binarized = closing * gammacorrected
    canvas=np.ones((3*TH,3*TW),dtype=np.uint8)*255
    canvas[TH:2*TH,TW:2*TW]=pseudo_binarized
    points=[]
    count = 0 # Tro add
    while(len(points)<1):
        count += 1 # Tro add
        if count > 50: # Tro add
            break # Tro add

        # Random shear
        shear_angle = np.random.uniform(param_min_shear, param_max_shear)
        M = np.float32([[1, shear_angle, 0], [0, 1, 0]])
        sheared = cv2.warpAffine(canvas, M, (3*TW, 3*TH), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST, borderValue=255)

        # Random rotation
        M = cv2.getRotationMatrix2D((3*TW/2, 3*TH/2), np.random.uniform(-param_rotation, param_rotation), 1)
        rotated = cv2.warpAffine(sheared, M, (3*TW, 3*TH), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST, borderValue=255)

        '''# Random scaling
        scaling_factor = np.random.uniform(1-param_scale, 1+param_scale)
        scaled = cv2.resize(rotated, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_NEAREST)'''

        scaled = rotated

        # detect cropping parameters
        points = np.argwhere(scaled!=255)
        points = np.fliplr(points)

    if len(points) < 1: # Tro add
        return pseudo_binarized

    r = cv2.boundingRect(np.array([points]))

    #random cropping
    deltax=random.randint(-param_movement_BB,param_movement_BB)
    deltay=random.randint(-param_movement_BB,param_movement_BB)
    x1=min(scaled.shape[0]-1,max(0,r[1]+deltax))
    y1=min(scaled.shape[1]-1,max(0,r[0]+deltay))
    x2=min(scaled.shape[0],x1+r[3])
    y2=min(scaled.shape[1],y1+r[2])
    final_image=np.uint8(scaled[x1:x2,y1:y2])

    return final_image

def randomize_labels(labels, batch_size, n_categories):
    number_of_changed_labels = random.choice(range(batch_size))
    indices_from_batch_changed = random.sample(range(batch_size), number_of_changed_labels)
    for i in indices_from_batch_changed:
        labels[i][0] = random.choice(range(n_categories))
    return labels

def normalize(tar):
    min_val = tar.min()
    max_val = tar.max()

    if max_val != min_val:
        tar = (tar - min_val) / (max_val - min_val)
    else:
        tar = np.zeros_like(tar)  # Asigna una matriz de ceros si min y max son iguales

    tar = tar * 255
    tar = tar.astype(np.uint8)
    return tar


def fine(label_list):
    if type(label_list) != list:
        return [label_list]
    else:
        return label_list

def write_image(xg, pred_label, gt_img, gt_label, title, iter):
    folder = '/data/gasbert/imagesGerard_handwritten'
    folder1 = folder + '/comparingImages'
    folder2 = folder + '/singleImages'
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folder1):
        os.makedirs(folder1)
    if not os.path.exists(folder2):
        os.makedirs(folder2)
    for cat in TARGET_CLASSES:
        if not os.path.exists(folder2 + "/" + cat):
            os.makedirs(folder2 + "/" + cat)
    batch_size = gt_label.shape[0]
    gt_img = gt_img.cpu().numpy()
    xg = xg.cpu().numpy()
    gt_label = gt_label.cpu().numpy()

    probs = nn.functional.softmax(pred_label, dim=1)

    pred_label = torch.topk(probs, 1, dim=-1)[1].squeeze(-1)  # b,t,83 -> b,t,1 -> b,t

    max_prob_per_row = torch.max(probs, dim=1)
    '''print("Largest prob: ")
    print(max_prob_per_row)'''

    pred_label = pred_label.cpu().numpy()

    outs = list()
    for i in range(batch_size):
        gt = gt_img[i].squeeze()
        tar = xg[i].squeeze()
        gt = normalize(gt)
        tar = normalize(tar)
        gt_text = gt_label[i].tolist()
        pred_text = pred_label[i].tolist()

        gt_text = fine(gt_text)
        pred_text = fine(pred_text)

        '''for j in range(num_tokens):
            gt_text = list(filter(lambda x: x != j, gt_text))
            pred_text = list(filter(lambda x: x != j, pred_text))'''
        
        gt_text_str = ''.join([index2letter[c] for c in gt_text])
        pred_text_str = ''.join([index2letter[c] for c in pred_text])
        #print(f"title: {title}")
        '''print('--')
        print(f"GT Text: {gt_text_str}")
        print(f"Predicted Text: {pred_text_str}")'''
        '''print("Class number: ")
        print(gt_text[0])
        print("Pred number: ")
        print(max_prob_per_row[1][i].item())
        print("Best prob: ")
        print(max_prob_per_row[0][i].item())'''
        if not (gt_label == 4).any().item():
            if(gt_text[0] == max_prob_per_row[1][i].item()):
                cv2.imwrite(folder2 + '/' + gt_text_str  + '/' + title + '_' + str(max_prob_per_row[0][i].item())[:4] + '.jpg', tar)

        gt_text = ''.join([index2letter[c] for c in gt_text])
        pred_text = ''.join([index2letter[c] for c in pred_text])
        gt_text_img = np.zeros_like(tar)
        pred_text_img = np.zeros_like(tar)
        cv2.putText(gt_text_img, gt_text_str, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(pred_text_img, pred_text_str, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out = np.vstack([gt, gt_text_img, tar, pred_text_img])
        outs.append(out)
    final_out = np.hstack(outs)
    if not (gt_label == 4).any().item() and iter % 29 == 0:
        cv2.imwrite(folder1 + '/' + title + '.jpg', final_out)

def write_image_cosine(xg, pred_label, gt_img, gt_label, iter, mean_cosine, mean_euclidean, mean_ssim):
    folder = '../../../../data/gasbert/imagesGerard_handwritten'
    folder1 = folder + '/cosine_images'
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folder1):
        os.makedirs(folder1)
    
    batch_size = gt_label.shape[0]
    gt_img = gt_img.cpu().numpy()
    xg = xg.cpu().numpy()
    gt_label = gt_label.cpu().numpy()

    probs = nn.functional.softmax(pred_label, dim=1)

    pred_label = torch.topk(probs, 1, dim=-1)[1].squeeze(-1)  # b,t,83 -> b,t,1 -> b,t

    pred_label = pred_label.cpu().numpy()

    outs = list()
    for i in range(batch_size):
        gt = gt_img[i].squeeze()
        tar = xg[i].squeeze()
        gt = normalize(gt)
        tar = normalize(tar)
        gt_text = gt_label[i].tolist()
        pred_text = pred_label[i].tolist()

        gt_text = fine(gt_text)
        pred_text = fine(pred_text)
        
        gt_text_str = ''.join([index2letter[c] for c in gt_text])
        pred_text_str = ''.join([index2letter[c] for c in pred_text])

        gt_text = ''.join([index2letter[c] for c in gt_text])
        pred_text = ''.join([index2letter[c] for c in pred_text])
        gt_text_img = np.zeros_like(tar)
        pred_text_img = np.zeros_like(tar)
        cv2.putText(gt_text_img, gt_text_str, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(pred_text_img, pred_text_str, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out = np.vstack([gt, gt_text_img, tar, pred_text_img])
        outs.append(out)
    final_out = np.hstack(outs)
    if not (gt_label == 4).any().item() and iter % 5 == 0:
        cv2.imwrite(folder1 + '/cosine_' + str(mean_cosine)[:5] + '-euclidean_' + str(mean_euclidean)[:5] + '-ssim_' + str(mean_ssim)[:6] + '.jpg', final_out)

def write_final_images(xg, pred_label, gt_img, gt_label, title, final_folder):
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)
    for cat in TARGET_CLASSES:
        if not os.path.exists(final_folder + "/" + cat):
            os.makedirs(final_folder + "/" + cat)
    batch_size = gt_label.shape[0]
    gt_img = gt_img.cpu().numpy()
    xg = xg.cpu().numpy()
    gt_label = gt_label.cpu().numpy()

    probs = nn.functional.softmax(pred_label, dim=1)

    pred_label = torch.topk(probs, 1, dim=-1)[1].squeeze(-1)  # b,t,83 -> b,t,1 -> b,t

    pred_label = pred_label.cpu().numpy()

    for i in range(batch_size):
        gt = gt_img[i].squeeze()
        tar = xg[i].squeeze()
        gt = normalize(gt)
        tar = normalize(tar)
        gt_text = gt_label[i].tolist()
        pred_text = pred_label[i].tolist()

        gt_text = fine(gt_text)
        pred_text = fine(pred_text)

        '''for j in range(num_tokens):
            gt_text = list(filter(lambda x: x != j, gt_text))
            pred_text = list(filter(lambda x: x != j, pred_text))'''
        
        gt_text_str = ''.join([index2letter[c] for c in gt_text])

        #Que no ho fagi amb les classes bad
        cv2.imwrite(final_folder + '/' + gt_text_str  + '/' + title + '.jpg', tar)



def return_image(xg, pred_label, gt_img, gt_label, title, iter):
    folder = '../../../../data/gasbert/imagesGerard_handwritten'
    folder1 = folder + '/comparingImages'
    folder2 = folder + '/singleImages'
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folder1):
        os.makedirs(folder1)
    if not os.path.exists(folder2):
        os.makedirs(folder2)
    for cat in TARGET_CLASSES:
        if not os.path.exists(folder2 + "/" + cat):
            os.makedirs(folder2 + "/" + cat)
    batch_size = gt_label.shape[0]
    gt_img = gt_img.cpu().numpy()
    xg = xg.cpu().numpy()
    gt_label = gt_label.cpu().numpy()

    probs = nn.functional.softmax(pred_label, dim=1)

    pred_label = torch.topk(probs, 1, dim=-1)[1].squeeze(-1)  # b,t,83 -> b,t,1 -> b,t

    max_prob_per_row = torch.max(probs, dim=1)
    '''print("Largest prob: ")
    print(max_prob_per_row)'''

    pred_label = pred_label.cpu().numpy()

    outs = list()
    for i in range(batch_size):
        gt = gt_img[i].squeeze()
        tar = xg[i].squeeze()
        gt = normalize(gt)
        tar = normalize(tar)
        gt_text = gt_label[i].tolist()
        pred_text = pred_label[i].tolist()

        gt_text = fine(gt_text)
        pred_text = fine(pred_text)

        gt_text_str = ''.join([index2letter[c] for c in gt_text])
        pred_text_str = ''.join([index2letter[c] for c in pred_text])
                
        gt_text = ''.join([index2letter[c] for c in gt_text])
        pred_text = ''.join([index2letter[c] for c in pred_text])
        gt_text_img = np.zeros_like(tar)
        pred_text_img = np.zeros_like(tar)
        cv2.putText(gt_text_img, gt_text_str, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(pred_text_img, pred_text_str, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out = np.vstack([gt, gt_text_img, tar, pred_text_img])
        outs.append(out)
    return np.hstack(outs)
    

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]

def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params

class DisModel(nn.Module):
    def __init__(self):
        super(DisModel, self).__init__()
        self.n_layers = 6
        self.final_size = 1024
        nf = 16
        cnn_f = [Conv2dBlock(1, nf, 7, 1, padding=3, pad_type='reflect', norm='none', activation='none')]
        for i in range(self.n_layers - 1):
            nf_out = min(nf * 2, 1024)
            cnn_f += [ActFirstResBlock(nf, nf, norm='none', activation='lrelu')]
            cnn_f += [ActFirstResBlock(nf, nf_out, norm='none', activation='lrelu')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2, padding=1)]
            nf = min(nf * 2, 1024)
        nf_out = min(nf * 2, 1024)
        cnn_f += [ActFirstResBlock(nf, nf, norm='none', activation='lrelu')]
        cnn_f += [ActFirstResBlock(nf, nf_out, norm='none', activation='lrelu')]

        self.cnn_f = nn.Sequential(*cnn_f)

        example_input = torch.randn(1, 1, 128, 128).to(device)  # Mover example_input al dispositivo correcto
        example_feat = self.cnn_f(example_input)
        flattened_size = np.prod(example_feat.shape[1:])

        #print(f"Flattened size: {flattened_size}")

        cnn_c = [
            nn.Flatten(),
            nn.Linear(flattened_size, self.final_size),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(p=0.3)
        ]
        self.cnn_c = nn.Sequential(*cnn_c)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        #print(f"x.shape in forward DisModel: {x.shape}")
        feat = self.cnn_f(x.to(device))
        #print(f"feat.shape after cnn_f: {feat.shape}")
        out = self.cnn_c(feat)
        #print(f"out.shape after cnn_c: {out.shape}")
        return out

    def calc_dis_fake_loss(self, input_fake):
        #print(f"input_fake.shape: {input_fake.shape}")
        label = torch.zeros(input_fake.shape[0], self.final_size).to(device)
        resp_fake = self.forward(input_fake)
        fake_loss = self.bce(resp_fake, label)
        return fake_loss

    def calc_dis_real_loss(self, input_real):
        #print(f"input_real.shape: {input_real.shape}")
        label = torch.ones(input_real.shape[0], self.final_size).to(device)
        resp_real = self.forward(input_real)
        real_loss = self.bce(resp_real, label)
        return real_loss

    def calc_gen_loss(self, input_fake):
        label = torch.ones(input_fake.shape[0], self.final_size).to(device)
        resp_fake = self.forward(input_fake)
        fake_loss = self.bce(resp_fake, label)
        return fake_loss

    def gradient_penalty(self, real_data, generated_data, reduction_factor=0.3):   
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        interpolated = epsilon * real_data + (1 - epsilon) * generated_data
        interpolated.requires_grad_(True)
        d_interpolated = self.forward(interpolated)
        grad_outputs = torch.ones_like(d_interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty * reduction_factor


class GenModel_FC(nn.Module):
    def __init__(self, w_noise, encoder_type = 'base', decoder_type = 'base'):
        super(GenModel_FC, self).__init__()
        if encoder_type == 'base':
            self.enc_image = ImageEncoder().to(device)
        elif encoder_type == 'multiple_level':
            self.enc_image = AggregatedImageEncoder().to(device)

        if decoder_type == 'base':
            self.dec = Decoder().to(device)
        elif decoder_type == 'multiple_level':
            self.dec = UpgradedDecoder().to(device)
        
        self.noise_std = w_noise

    def decode(self, content):
        images = self.dec(content)
        return images

    def forward(self, img, label, return_encoding=False):
        feat_xs = self.enc_image(img.to(device))  # Codifica la imagen de entrada
        #print(f"Encoded features shape: {feat_xs.shape}")
        if return_encoding:
            return feat_xs
        #Add random noise to encoded representation of the image
        noise = torch.randn_like(feat_xs).to(device) * self.noise_std
        feat_xs_noisy = feat_xs + noise

        # Add label to generator decoder input
        # Change when increasing the number of categories
        shp = list(label.shape)
        lst = []
        if(len(shp) == 1):
            #values = torch.tensor([int(label[0]), int(label[1]), int(label[2]), int(label[3])]).view(4, 1, 1, 1)
            for clss in label:
                tensor = torch.full((len(TARGET_CLASSES), 4, 4), 0.)
                tensor[clss] = torch.full((4, 4), 1.)
                lst.append(tensor)
            
        elif(len(shp) == 2):
            #values = torch.tensor([int(label[0][0]), int(label[1][0]), int(label[2][0]), int(label[3][0])]).view(4, 1, 1, 1)
            for clss in label:
                tensor = torch.full((len(TARGET_CLASSES), 4, 4), 0.)
                tensor[clss[0]] = torch.full((4, 4), 1.)
                lst.append(tensor)
        
        # Concatenate the tensors along a new dimension (0) to form a 4xNCLASSESx4x4 tensor
        result = torch.stack(lst, dim=0).to(device)      

        #additional_feature = values.expand(-1, 15, 4, 4).to(device)
        new_tensor = torch.cat((feat_xs_noisy, result), dim=1)

        #print(f"New tensor shape: {new_tensor.shape}")
        generated_img = self.decode(new_tensor)  # Decodifica para generar la imagen
        #print(f"Generated image shape: {generated_img.shape}")

        min_vals = torch.amin(generated_img, dim=(2, 3), keepdim=True)  # Shape: [B, C, 1, 1]
        max_vals = torch.amax(generated_img, dim=(2, 3), keepdim=True)  # Shape: [B, C, 1, 1]
        
        # Avoid division by zero by adding a small epsilon
        generated_img = (generated_img - min_vals) / (max_vals - min_vals + 1e-8)

        return generated_img

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = vgg19_bn(False).to(device)
        self.output_dim = 512
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.model(x.to(device))
        x = self.dropout(x)
        return x


class MultipleLevel_ImageEncoder(nn.Module):
    def __init__(self):
        super(MultipleLevel_ImageEncoder, self).__init__()
        vgg = vgg19_bn(pretrained=False)
        self.device = device

        # Split VGG into blocks for multi-scale feature extraction
        self.stage1 = vgg.features[:6]    # Captures low-level features (e.g., edges)
        self.stage2 = vgg.features[6:13]  # Captures mid-level features
        self.stage3 = vgg.features[13:26] # Captures high-level features
        self.stage4 = vgg.features[26:]   # Higher-level features with more abstraction

        self.dropout = nn.Dropout(p=0.3)
        self.output_dim = 512  # Can be modified based on your needs

    def forward(self, x):
        x = x.to(self.device)

        # Forward pass through each stage
        x1 = self.stage1(x)  # Low-level
        x2 = self.stage2(x1)  # Mid-level
        x3 = self.stage3(x2)  # Higher-level
        x4 = self.stage4(x3)  # Highest-level

        # Apply dropout to the last stage
        x4 = self.dropout(x4)

        # Return all stages for multi-scale aggregation
        return x1, x2, x3, x4


class AggregatedImageEncoder(nn.Module):
    def __init__(self):
        super(AggregatedImageEncoder, self).__init__()
        self.encoder = MultipleLevel_ImageEncoder()
        self.encoder = self.encoder.to(self.encoder.device)
        # Convolutions to reduce feature dimensions and merge scales
        self.conv1 = nn.Conv2d(64, 128, kernel_size=1)   # Stage 1 (low-level)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=1)  # Stage 2 (mid-level)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=1)  # Stage 3 (higher-level)
        self.conv4 = nn.Conv2d(512, 128, kernel_size=1)  # Stage 4 (highest-level)

        # Final layer to combine all scales
        self.final_conv = nn.Conv2d(128 * 4, self.encoder.output_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # Get multi-scale features
        x1, x2, x3, x4 = self.encoder(x)

        # Reduce dimensionality of each feature map
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        # Upsample to the same spatial dimensions (if needed)
        x2 = nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x3 = nn.functional.interpolate(x3, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x4 = nn.functional.interpolate(x4, size=x1.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate along the channel dimension
        x_aggregated = torch.cat([x1, x2, x3, x4], dim=1)

        # Final convolution to merge features and reduce dimension
        output = self.final_conv(x_aggregated)
        output = nn.functional.interpolate(output, size=(4, 4), mode='bilinear', align_corners=False)

        return output


class Decoder(nn.Module):
    def __init__(self, ups=5, n_res=2, dim= 512+len(TARGET_CLASSES), out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, "none", activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [
                nn.Upsample(scale_factor=2),  # Duplicar resolución en cada paso
                Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='in', activation=activ, pad_type=pad_type),
                nn.Dropout(p=0.3)
            ]
            dim //= 2
        self.model += [
            Conv2dBlock(dim, out_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for layer in self.model:
            x = layer(x.to(device))
            #print(f"Layer output shape: {x.shape}")  # Imprime el tamaño de salida de cada capa
        return x


class UpgradedDecoder(nn.Module):
    def __init__(self, ups=5, n_res=4, dim=512+len(TARGET_CLASSES), out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super(UpgradedDecoder, self).__init__()

        # Multi-level processing similar to encoder
        self.model = []
        
        # Increased number of residual blocks
        self.model += [ResBlocks(n_res, dim, "none", activ, pad_type=pad_type)]

        for i in range(ups):
            self.model += [
                nn.Upsample(scale_factor=2),  # Double resolution at each step
                Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='in', activation=activ, pad_type=pad_type),
                nn.Dropout(p=0.3)
            ]
            dim //= 2
        
        # Additional convolutional layer after upsampling for more refinement
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm='in', activation=activ, pad_type=pad_type)]
        
        # Final convolutional layer
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        
        # Sequential model
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for layer in self.model:
            x = layer(x.to(device))
        return x

class RecModel_seq2seq(nn.Module):
    def __init__(self, vocab_size, pretrained=False):
        super(RecModel_seq2seq, self).__init__()
        hidden_size_enc = hidden_size_dec = 512
        embed_size = 60
        self.enc = rec_encoder(hidden_size_enc, IMG_HEIGHT, IMG_WIDTH, True, None, False).to(device)
        self.dec = rec_decoder(hidden_size_dec, embed_size, vocab_size, rec_attention, None).to(device)
        self.seq2seq = rec_seq2seq(self.enc, self.dec, vocab_size).to(device)
        if pretrained:
            model_file = "/home/gasbert/Desktop/GANGerard/save_weights/recognizer-2-SEQ2SEQ.model"
                      

            self.load_state_dict(torch.load(model_file))
            self.eval()

    def forward(self, img, label, img_width):
        self.seq2seq.train()
        img = torch.cat([img, img, img], dim=1).to(device)  # b,1,64,128->b,3,64,128
        output, attn_weights = self.seq2seq(img, label.to(device), img_width.to(device), teacher_rate=False, train=False)
        return output.permute(1, 0, 2)  # t,b,83->b,t,83


class RecModel_CNN(nn.Module):
    def __init__(self, vocab_size, pretrained=False):
        super(RecModel_CNN, self).__init__()
        self.pretrained = pretrained
        # CNN Architecture based on the previous example
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input is 3 channels due to RGB concatenation in your previous code
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Flatten
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  # Adjust based on input image size after convolution
        self.fc2 = nn.Linear(128, vocab_size)  # vocab_size output classes

        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3_bn = nn.BatchNorm2d(128)
        

        if self.pretrained:
            model_file = "/home/gasbert/Desktop/GANGerard/save_weights/recognizer_fornes-4-1.0.model"
            self.load_state_dict(torch.load(model_file))
            self.eval()

    def forward(self, img, label, img_width, return_features=False):
        # Replace the concatenation logic (b,1,64,128 -> b,3,64,128) from your previous code
        img = torch.cat([img, img, img], dim=1).to(device)  # Assuming input is grayscale and you're converting to 3 channels
        
        # Pass through CNN layers
        x = self.pool1(nn.functional.relu(self.conv1_bn(self.conv1(img))))
        x = self.pool2(nn.functional.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool3(nn.functional.relu(self.conv3_bn(self.conv3(x))))

        #print(f"Shape before flattening: {x.shape}")

        # Flatten
        x = x.view(-1, 128 * 16 * 16)
        #x = x.view(x.size(0), -1)

        #print(f"Shape after flattening: {x.shape}")

        if return_features:
            return x # Return features for FID

        # Fully connected layers
        x = nn.functional.relu(self.fc1(x))
        output = self.fc2(x)

        return output  # Output is (batch_size, vocab_size)


class MLP(nn.Module):
    def __init__(self, in_dim=64, out_dim=4096, dim=256, n_blk=3, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1).to(device))

if __name__ == "__main__":
    # Prueba del generador
    gen_model = GenModel_FC().to(device)
    sample_img = torch.randn(128, 1, 128, 128).to(device)  # Ejemplo de entrada
    generated_img = gen_model(sample_img)
    #print(f"Generated image shape: {generated_img.shape}")

    # Prueba del discriminador
    dis_model = DisModel().to(device)
    output = dis_model(generated_img)
    #print(f"Output shape: {output.shape}")

