import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


file_path = './all_images.txt'

img_format = '.bmp'


IMG_HEIGHT = 128
IMG_WIDTH = 128

test_images_per_symbol = 100


tokens = {
    #'PAD_TOKEN': 0,
    #'SOS_TOKEN': 1,
    #'EOS_TOKEN': 2,
    #'UNK_TOKEN': 3
}

#tokens = {}

# Clases de interés
TARGET_CLASSES = sorted({'accidentalsharp', 'notewhole', 'barline',
        'quarternotedown', 'halfnotedown', 'legerline', 'beam', 'tieslur',
        'accidentalnatural', 'dot', 'accidentaldoublesharp', 'halfwholerest', 
        'halfnoteup', 'accidentalflat', 'quarternoteup', 'eighthnotedown', 
        'gclef', 'measureseparator', 'cclefb', 'timesigcommon', 'cuttime', 
        'eighthrest', 'eighthnoteup', 'fclef', 'quarterrest', 'cclefk',
        'sixteenthrest', 'sixteenthnoteup', 'articulationstaccato', 
        'sixteenthnotedown', 'timesig34', 'timesig24', 'timesig68', 
        'thirtytwonoteup', 'sixtyfourrest', 'timesig98', 'timesig22', 
        'timesig44', 'timesig38', 'sixtyfournotedown', 'thirtytwonotedown', 
        'timesig128', 'thirtytworest', 'sixtyfournoteup', 'flagup', 'flagdown', 
        'noteheadfull', 'stem', 'tie', 'accidentalsharpbad', 'gclefbad'})

'''TARGET_CLASSES = sorted({'accidentalsharp', 'accidentalflat', 
        'gclef', 'fclef'})'''

IMPORTANT_CLASSES = sorted({'accidentalsharp', 'gclef'})

#TARGET_CLASSES = sorted({'accidentalsharp', 'quarternotedown', 'wholenote', 'halfnotedown'})

# Agregar las clases de interés a los tokens
for i, symbol in enumerate(sorted(TARGET_CLASSES)):
    tokens[symbol] = i

onehotencoder = dict()
nCat = len(TARGET_CLASSES)
for id, cat in enumerate(sorted(TARGET_CLASSES)):
    arr = np.zeros(nCat)
    arr[id] = 1
    onehotencoder[cat] = arr

index2letter = {v: k for k, v in tokens.items()}
vocab_size = len(tokens)
num_tokens = 4
NUM_CHANNEL = 3

print(f"vocab_size: {vocab_size}")

class MusicSymbolDataset(Dataset):
    def __init__(self, data_dirs, target_classes, transform=None, test=False):
        global tokens
        self.data_dirs = data_dirs
        self.target_classes = target_classes
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        self.data = []
        self.classes = []
        classes_dict = {key: 0 for key in self.target_classes}
        imgs_per_dataset = test_images_per_symbol/len(data_dirs)
        for data_dir in data_dirs:
            print(f"Procesando directorio: {data_dir}")
            if not os.path.exists(data_dir):
                print(f"Directorio no existe: {data_dir}")
                continue
            for symbol in sorted(os.listdir(data_dir)):
                symbol_index = 0
                symbol_dir = os.path.join(data_dir, symbol)
                symbol_lower = symbol.lower()
                if symbol_lower in self.target_classes and os.path.isdir(symbol_dir):
                    if symbol_lower not in tokens:
                        tokens[symbol_lower] = len(tokens)
                    if symbol_lower not in self.classes:
                        self.classes.append(symbol_lower)
                    print(f"Existente: {symbol_dir}")
                    png_count = 0
                    for img_file in os.listdir(symbol_dir):
                        if img_file.lower().endswith(img_format):
                            if symbol_index >= imgs_per_dataset and test:
                                break
                            png_count += 1
                            self.data.append((os.path.join(symbol_dir, img_file), tokens[symbol_lower]))
                            print(f"Añadido: {os.path.join(symbol_dir, img_file)}")
                            symbol_index += 1
                    print(f"Archivos .png encontrados en {symbol_dir}: {png_count}")
                else:
                    print(f"Clase no encontrada o no es un directorio: {symbol_lower}")
                
        print(f"Ejemplo data: {self.data[0]}")
        print(f"tokens: {tokens}")
        print(f"Total de imágenes encontradas: {len(self.data)}")
        print(f"Clases encontradas: {self.classes}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        return image, label

def loadData(directories=None, batch_size=128, num_workers=0, test_split_ratio=0.1):
    if directories is None:
        #directories = ['../Projecte_GANs/Datasets/Printed/deepscores_symbols_reduced'] #'./dataset1', './dataset2',
        directories = ['/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/CapitanSymbols_rot_flip',
                    '/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/Fornes_Dataset_rot_flip',
                    '/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/imatges_Homus_rot_flip',
                    '/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/Muscima_Symbols_rot_flip',
                    '/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/bad_symbols']

    dataset = MusicSymbolDataset(directories, TARGET_CLASSES)

    if len(dataset) == 0:
        raise ValueError("El dataset está vacío")

    # Dividir el dataset en entrenamiento y prueba
    test_size = int(test_split_ratio * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"\nlen(train_dataset): {len(train_dataset)}")
    print(f"\nlen(test_dataset): {len(test_dataset)}")
    print(f"\nindex2letter: {index2letter}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def loadData_imp(directories=None, batch_size=128, num_workers=0):
    if directories is None:
        #directories = ['../Projecte_GANs/Datasets/Printed/deepscores_symbols_reduced'] #'./dataset1', './dataset2',
        directories = ['/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/CapitanSymbols_rot_flip',
                    '/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/Fornes_Dataset_rot_flip',
                    '/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/imatges_Homus_rot_flip',
                    '/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/Muscima_Symbols_rot_flip',
                    '/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/bad_symbols']

    dataset = MusicSymbolDataset(directories, IMPORTANT_CLASSES)
        
    if len(dataset) == 0:
        raise ValueError("El dataset está vacío")

    
    imp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return imp_loader


def loadData_sample(directories=None, batch_size=128, num_workers=0):
    if directories is None:
        #directories = ['../Projecte_GANs/Datasets/Printed/deepscores_symbols_reduced'] #'./dataset1', './dataset2',
        directories = ['/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/generate_sample_dataset']

    dataset = MusicSymbolDataset(directories, TARGET_CLASSES)
        
    if len(dataset) == 0:
        raise ValueError("El dataset está vacío")

    
    imp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return imp_loader

def loadData_generate(directories=None, batch_size=128, num_workers=0):
    if directories is None:
        #directories = ['../Projecte_GANs/Datasets/Printed/deepscores_symbols_reduced'] #'./dataset1', './dataset2',
        directories = ['/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/CapitanSymbols_rot_flip',
                    '/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/Fornes_Dataset_rot_flip',
                    '/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/imatges_Homus_rot_flip',
                    '/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/Muscima_Symbols_rot_flip',
                    '/home/gasbert/Desktop/Projecte_GANs/Datasets/Handwritten/datasets_rot_flip/bad_symbols']

    dataset = MusicSymbolDataset(directories, TARGET_CLASSES, test=True)

    if len(dataset) == 0:
        raise ValueError("El dataset está vacío")

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader
