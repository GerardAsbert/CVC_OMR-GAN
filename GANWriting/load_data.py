import os
import math
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


file_path = './all_images.txt'

img_format = '.png'


IMG_HEIGHT = 128
IMG_WIDTH = 128

test_images_per_symbol = 100


# Earlier hard-coded class lists, kept for reference. The class list is now
# read from the dataset folders by set_data_root() below.
'''TARGET_CLASSES = sorted({'accidentalsharp', 'notewhole', 'barline',
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
        'noteheadfull', 'stem', 'tie', 'accidentalsharpbad', 'gclefbad'})'''

'''TARGET_CLASSES = sorted({'flag_up', 'flag_down', 'g_clef', 'notehead_empty', 'notehead_full', 
                            'quarter_rest', 'sharp', 'stem'})'''

IMPORTANT_CLASSES = sorted({'accidentalsharp', 'gclef'})

num_tokens = 4
NUM_CHANNEL = 3


# ── symbol classes, read from the dataset ─────────────────────────────────────
# set_data_root(<dataset root>) scans every author folder under the root and
# takes each symbol folder holding at least one image as a class: the union over
# authors, with the exact (case-sensitive) folder names, sorted. It must run
# BEFORE network_tro / modules_tro / loss_tro are imported, because they size the
# model from these values at import time. Until then the names in
# _CLASS_GLOBALS are undefined and reading them raises (see __getattr__).
_CLASS_GLOBALS = ('TARGET_CLASSES', 'tokens', 'index2letter', 'onehotencoder', 'nCat', 'vocab_size')
_data_root = None


def discover_classes(data_root):
    """Sorted union of the symbol folder names, over all author folders, that hold
    at least one image."""
    classes = set()
    for author in author_dirs(data_root):
        for symbol in os.listdir(author):
            symbol_dir = os.path.join(author, symbol)
            if os.path.isdir(symbol_dir) and any(f.lower().endswith(img_format)
                                                 for f in os.listdir(symbol_dir)):
                classes.add(symbol)
    assert classes, f"no symbol folders with {img_format} images found under {data_root!r}"
    return sorted(classes)


def set_data_root(data_root):
    """Fix the dataset root and build the class vocabulary from it. Only the first
    call takes effect; a later call with a different root is an error, since the
    model has already been sized from the first one."""
    global _data_root, TARGET_CLASSES, tokens, index2letter, onehotencoder, nCat, vocab_size
    root = os.path.abspath(data_root)
    if _data_root is not None:
        assert root == _data_root, (f"load_data classes were already read from {_data_root!r}; "
                                    f"cannot switch to {root!r} in the same process")
        return
    TARGET_CLASSES = discover_classes(root)
    tokens = {symbol: i for i, symbol in enumerate(TARGET_CLASSES)}
    index2letter = {i: symbol for symbol, i in tokens.items()}
    nCat = len(TARGET_CLASSES)
    onehotencoder = {symbol: np.eye(nCat)[i] for i, symbol in enumerate(TARGET_CLASSES)}
    vocab_size = len(tokens)
    _data_root = root
    print(f"[load_data] {vocab_size} classes read from {root}: {TARGET_CLASSES}")


def classes_configured():
    return _data_root is not None


def __getattr__(name):
    if name in _CLASS_GLOBALS:
        # RuntimeError, not AttributeError, so `from load_data import vocab_size`
        # shows this message instead of a bare "cannot import name".
        raise RuntimeError(f"load_data.{name} is not set yet: call "
                           f"load_data.set_data_root(<dataset root>) before importing "
                           f"network_tro / modules_tro / loss_tro.")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _resolve_class(name, classes):
    """`name` if it is in `classes`, else its unique case-insensitive match there
    (so e.g. IMPORTANT_CLASSES' 'gclef' still finds 'gClef'), else None. An
    ambiguous match (Copiale's 'B' vs 'b') resolves to None, never the wrong one."""
    if name in classes:
        return name
    matches = [c for c in classes if c.lower() == name.lower()]
    return matches[0] if len(matches) == 1 else None


# ── background / padding colour (mirrors the SNN's pad_color) ──────────────────
# The dataset background colour decides the fill used when padding a non-square
# symbol to a square canvas before resizing, exactly like the SNN pipeline.
BG_COLOR = 'white'   # 'white' -> pad with 255, 'black' -> pad with 0


def _bg_fill(bg_color):
    """PIL grayscale fill for the given background colour name."""
    bg = str(bg_color).lower()
    assert bg in ('black', 'white'), f"bg_color must be 'black' or 'white', got {bg_color!r}"
    return 255 if bg == 'white' else 0


def author_dirs(data_root):
    """The author folders inside `data_root`, sorted, as MusicSymbolDataset expects."""
    assert os.path.isdir(data_root), f"data_root does not exist: {data_root!r}"
    dirs = sorted(os.path.join(data_root, d) for d in os.listdir(data_root)
                  if os.path.isdir(os.path.join(data_root, d)))
    assert dirs, f"no author folders found in {data_root!r}"
    return dirs


def run_tag(data_root, dataset_percentage, subset_seed):
    """Per-run output subfolder, '<dataset folder name>/pct<P>_seed<S>', so runs on
    different datasets, percentages or seeds never write into the same place."""
    dataset_name = os.path.basename(os.path.normpath(data_root))
    return os.path.join(dataset_name, f"pct{dataset_percentage:g}_seed{subset_seed}")


def _select_subset(files, percentage, seed, author, symbol):
    """Seeded subset of one (author, symbol) folder's image files.

    percentage in [0, 100]: 100 keeps everything, 0 keeps exactly one image, and
    anything else keeps ceil(percentage% of the folder), never fewer than one.
    The files are permuted by an RNG seeded from (seed, author, symbol) alone, so
    the choice does not depend on os.listdir order or on which other folders are
    loaded, and a smaller percentage is always a subset of a larger one."""
    assert 0 <= percentage <= 100, f"dataset_percentage must be in [0, 100], got {percentage!r}"
    if percentage >= 100 or not files:
        return files
    k = 1 if percentage == 0 else max(1, math.ceil(len(files) * percentage / 100))
    order = sorted(files)
    random.Random(f"{seed}/{author}/{symbol}").shuffle(order)
    return sorted(order[:k])


def _pad_to_square(img, fill):
    """Pad a PIL image to a square canvas with `fill`, centred (SNN-style).

    Works for both 'L' and 'RGB' sources: the canvas keeps the input's mode, and
    the scalar `fill` is expanded to a per-channel tuple when needed, so the gray
    and colour views of the same file get exactly the same geometry."""
    w, h = img.size
    s = max(w, h)
    bands = len(img.getbands())
    canvas = Image.new(img.mode, (s, s), fill if bands == 1 else (fill,) * bands)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


class MusicSymbolDataset(Dataset):
    def __init__(self, data_dirs, target_classes=None, transform=None, test=False, bg_color=BG_COLOR,
                 return_author=False, return_color=False, dataset_percentage=100, subset_seed=0):
        if not classes_configured():
            raise RuntimeError("call load_data.set_data_root(<dataset root>) before building a dataset")
        target_classes = TARGET_CLASSES if target_classes is None else target_classes
        self.data_dirs = data_dirs
        self.target_classes = target_classes
        # When True, __getitem__ returns (image, label, author) instead of
        # (image, label). `author` is the top-level input folder name (the
        # copyist/author dir), so inference can mirror the dataset's
        # <author_name>/<symbol_name> layout on disk. Off by default so the
        # training loaders keep their original (image, label) contract.
        self.return_author = return_author
        # When True, the ORIGINAL colour version of the same file is appended as
        # the LAST element of the returned tuple: (image, label, color) or
        # (image, label, author, color). It is purely a metric target — the model
        # never sees it — used to log the MSE against the true colour image
        # alongside the grayscale one. Off by default (it costs a second decode).
        self.return_color = return_color
        # Seeded per-(author, symbol) subsampling, see _select_subset. Applied
        # before the `test` cap, so inference only sees images from the subset.
        self.dataset_percentage = dataset_percentage
        self.subset_seed = subset_seed
        self.pad_fill = _bg_fill(bg_color)   # square-padding fill from background colour
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        # Same geometry as `self.transform`, minus the grayscale step, so the
        # colour target stays pixel-aligned with the grayscale one.
        self.color_transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
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
            # The author/copyist folder name (== the input dir name), used to
            # rebuild the <author_name>/<symbol_name> tree at inference time.
            author = os.path.basename(os.path.normpath(data_dir))
            for symbol in sorted(os.listdir(data_dir)):
                symbol_index = 0
                symbol_dir = os.path.join(data_dir, symbol)
                # Match the folder to a requested class, then spell it the way
                # TARGET_CLASSES does so it maps onto the model's label index.
                symbol_key = _resolve_class(symbol, self.target_classes)
                if symbol_key is not None:
                    symbol_key = _resolve_class(symbol_key, TARGET_CLASSES) or symbol_key
                if symbol_key is not None and os.path.isdir(symbol_dir):
                    if symbol_key not in tokens:
                        tokens[symbol_key] = len(tokens)
                    if symbol_key not in self.classes:
                        self.classes.append(symbol_key)
                    print(f"Existente: {symbol_dir}")
                    png_count = 0
                    img_files = [f for f in os.listdir(symbol_dir) if f.lower().endswith(img_format)]
                    img_files = _select_subset(img_files, self.dataset_percentage, self.subset_seed,
                                               author, symbol_key)
                    for img_file in img_files:
                        if symbol_index >= imgs_per_dataset and test:
                            break
                        png_count += 1
                        self.data.append((os.path.join(symbol_dir, img_file), tokens[symbol_key], author))
                        print(f"Añadido: {os.path.join(symbol_dir, img_file)}")
                        symbol_index += 1
                    print(f"Archivos .png encontrados en {symbol_dir}: {png_count}")
                else:
                    print(f"Clase no encontrada o no es un directorio: {symbol}")
                
        if not self.data:
            raise ValueError(f"no images found for classes {self.target_classes} in {data_dirs}")
        print(f"Ejemplo data: {self.data[0]}")
        print(f"tokens: {tokens}")
        print(f"Total de imágenes encontradas: {len(self.data)}")
        print(f"Clases encontradas: {self.classes}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, author = self.data[idx]
        src = Image.open(img_path)
        image = src.convert('L')
        image = _pad_to_square(image, self.pad_fill)   # square-pad with bg colour, then resize
        image = self.transform(image)

        out = (image, label, author) if self.return_author else (image, label)
        if self.return_color:
            # Same pad-then-resize path, kept in RGB: (3, IMG_HEIGHT, IMG_WIDTH).
            color = _pad_to_square(src.convert('RGB'), self.pad_fill)
            out = out + (self.color_transform(color),)
        return out

def loadData(directories=None, batch_size=128, num_workers=0, test_split_ratio=0.1, bg_color=BG_COLOR,
             return_color=False, dataset_percentage=100, subset_seed=0):
    if directories is None:
        #directories = ['../Projecte_GANs/Datasets/Printed/deepscores_symbols_reduced'] #'./dataset1', './dataset2',
        directories = ['/data/113-2/users/gasbert/TFM_SNN/UFAL.OmniOMR_train_SymbolLevel_3auth_8inst/manuscript copy, copyist: I. B.',
                       '/data/113-2/users/gasbert/TFM_SNN/UFAL.OmniOMR_train_SymbolLevel_3auth_8inst/manuscript copy, copyist: Zirnstein, Anton (19th century)',
                       '/data/113-2/users/gasbert/TFM_SNN/UFAL.OmniOMR_train_SymbolLevel_3auth_8inst/manuscript copy, copyist: Schicht, Johann Gottfried (1753-1823)']

    dataset = MusicSymbolDataset(directories, bg_color=bg_color,
                                 return_color=return_color,
                                 dataset_percentage=dataset_percentage, subset_seed=subset_seed)

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


def loadData_full(directories=None, batch_size=128, num_workers=0, bg_color=BG_COLOR,
                  return_color=False, dataset_percentage=100, subset_seed=0):
    """Same as loadData, but with NO train/test split: the whole dataset is used
    both for training and for evaluation. The two loaders wrap the SAME dataset
    object (built once, so the directory scan only happens once); only the
    shuffling differs, so the eval pass keeps a fixed, reproducible order.

    Note that with this loader 'curve/eval_mse' measures fit on the training
    data, not generalisation."""
    if directories is None:
        directories = ['/data/113-2/users/gasbert/TFM_SNN/UFAL.OmniOMR_train_SymbolLevel_3auth_8inst/manuscript copy, copyist: I. B.',
                       '/data/113-2/users/gasbert/TFM_SNN/UFAL.OmniOMR_train_SymbolLevel_3auth_8inst/manuscript copy, copyist: Zirnstein, Anton (19th century)',
                       '/data/113-2/users/gasbert/TFM_SNN/UFAL.OmniOMR_train_SymbolLevel_3auth_8inst/manuscript copy, copyist: Schicht, Johann Gottfried (1753-1823)']

    dataset = MusicSymbolDataset(directories, bg_color=bg_color,
                                 return_color=return_color,
                                 dataset_percentage=dataset_percentage, subset_seed=subset_seed)

    if len(dataset) == 0:
        raise ValueError("El dataset está vacío")

    print(f"\nlen(train_dataset): {len(dataset)}  (dataset completo, sin split)")
    print(f"\nlen(test_dataset): {len(dataset)}  (las MISMAS instancias que train)")
    print(f"\nindex2letter: {index2letter}")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def loadData_imp(directories=None, batch_size=128, num_workers=0, bg_color=BG_COLOR):
    if directories is None:
        #directories = ['../Projecte_GANs/Datasets/Printed/deepscores_symbols_reduced'] #'./dataset1', './dataset2',
        directories = ['/data2/users/gasbert/music-symbol-GAN/symbol-datasets/Handwritten/datasets_rot_flip/CapitanSymbols_rot_flip',
                    '/data2/users/gasbert/music-symbol-GAN/symbol-datasets/Handwritten/datasets_rot_flip/Fornes_Dataset_rot_flip',
                    '/data2/users/gasbert/music-symbol-GAN/symbol-datasets/Handwritten/datasets_rot_flip/imatges_Homus_rot_flip',
                    '/data2/users/gasbert/music-symbol-GAN/symbol-datasets/Handwritten/datasets_rot_flip/Muscima_Symbols_rot_flip',
                    '/data2/users/gasbert/music-symbol-GAN/symbol-datasets/Handwritten/datasets_rot_flip/bad_symbols']

    dataset = MusicSymbolDataset(directories, IMPORTANT_CLASSES, bg_color=bg_color)
        
    if len(dataset) == 0:
        raise ValueError("El dataset está vacío")

    
    imp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return imp_loader


def loadData_sample(directories=None, batch_size=128, num_workers=0, bg_color=BG_COLOR):
    if directories is None:
        #directories = ['../Projecte_GANs/Datasets/Printed/deepscores_symbols_reduced'] #'./dataset1', './dataset2',
        directories = ['/data2/users/gasbert/music-symbol-GAN/symbol-datasets/Handwritten/generate_sample_dataset']

    dataset = MusicSymbolDataset(directories, bg_color=bg_color)
        
    if len(dataset) == 0:
        raise ValueError("El dataset está vacío")

    
    imp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return imp_loader

def loadData_generate(directories=None, batch_size=128, num_workers=0, bg_color=BG_COLOR,
                      dataset_percentage=100, subset_seed=0):
    if directories is None:
        #directories = ['../Projecte_GANs/Datasets/Printed/deepscores_symbols_reduced'] #'./dataset1', './dataset2',
        directories = ['/data/113-2/users/gasbert/TFM_SNN/UFAL.OmniOMR_train_SymbolLevel_3auth_8inst/manuscript copy, copyist: I. B.',
                       '/data/113-2/users/gasbert/TFM_SNN/UFAL.OmniOMR_train_SymbolLevel_3auth_8inst/manuscript copy, copyist: Zirnstein, Anton (19th century)',
                       '/data/113-2/users/gasbert/TFM_SNN/UFAL.OmniOMR_train_SymbolLevel_3auth_8inst/manuscript copy, copyist: Schicht, Johann Gottfried (1753-1823)']

    # Every image of the (subsampled) dataset, with no per-symbol cap, so
    # inference regenerates all the training samples.
    dataset = MusicSymbolDataset(directories, bg_color=bg_color,
                                 return_author=True,
                                 dataset_percentage=dataset_percentage, subset_seed=subset_seed)

    if len(dataset) == 0:
        raise ValueError("El dataset está vacío")

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader