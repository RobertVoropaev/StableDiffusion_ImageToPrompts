import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import warnings
warnings.filterwarnings('ignore')

import sys
import pickle
import json
import glob
import gc
import random
import time
import unicodedata
import traceback
import datetime
import copy
import argparse

import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from matplotlib import pyplot as plt 
from tqdm.notebook import tqdm
from pathlib import Path
from scipy.spatial import distance
from collections import defaultdict
from PIL import Image

from sklearn.model_selection import train_test_split

import torch
import torchvision
from torchvision.models import (
    vit_b_16, ViT_B_16_Weights, 
    vit_l_16, ViT_L_16_Weights,
    vit_h_14, ViT_H_14_Weights,
    regnet_y_32gf, RegNet_Y_32GF_Weights,
    regnet_y_128gf, RegNet_Y_128GF_Weights,
    regnet_y_16gf, RegNet_Y_16GF_Weights,
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    convnext_large, ConvNeXt_Large_Weights,
    swin_v2_b, Swin_V2_B_Weights
)
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, 
    ConstantLR, LinearLR, 
    ExponentialLR, PolynomialLR, 
    CosineAnnealingLR, CosineAnnealingWarmRestarts, 
    CyclicLR, OneCycleLR, 
    ReduceLROnPlateau
)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
######### batch config #########

batch_size_config = {
    "vit_b_16": {
        True: 256,
        False: 16
    },
    "vit_b_16_linear": {
        True: 256,
        False: 48
    },
    "vit_l_16": {
        True: 16,
        False: 1
    }, 
    "vit_h_14": {
        True: 4,
        False: 1
    },
    "regnet_y_16gf": {
        True: 64,
        False: 26
    },
    "regnet_y_16gf_linear": {
        True: 64,
        False: 26
    },
    "regnet_y_32gf": {
        True: 16,
        False: 6
    },
    "regnet_y_32gf_linear": {
        True: 16,
        False: 20
    },
    "regnet_y_128gf": {
        True: 4,
        False: 1
    },
    "regnet_y_128gf_linear": {
        True: 4,
        False: 1
    },
    "efficientnet_v2_l": {
        True: 64,
        False: 3
    },
    "efficientnet_v2_m": {
        True: 64,
        False: 6
    },
    "convnext_large": {
        True: 64,
        False: 12
    },
    "swin_v2_b": {
        True: 256,
        False: 16
    }

}

######### VAL FUNCTIONS #########

def get_img_model(img_model_name: str, load_weight: bool, head_emb_size: int):
    if img_model_name == "regnet_y_16gf":
        if not load_weight:
            model = regnet_y_16gf()
        else:
            weights = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
            model = regnet_y_16gf(weights=weights)
        model.fc = torch.nn.Linear(3024, head_emb_size)
        
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    if img_model_name == "regnet_y_16gf_linear":
        if not load_weight:
            model = regnet_y_16gf()
        else:
            weights = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1
            model = regnet_y_16gf(weights=weights)
        model.fc = torch.nn.Linear(3024, head_emb_size)
        
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif img_model_name == "regnet_y_32gf":
        if not load_weight:
            model = regnet_y_32gf()
        else:
            weights = RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1
            model = regnet_y_32gf(weights=weights)
        model.fc = torch.nn.Linear(3712, head_emb_size)
        
        preprocess = transforms.Compose([
            transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
        
    elif img_model_name == "regnet_y_32gf_linear":
        if not load_weight:
            model = regnet_y_32gf()
        else:
            weights = RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1
            model = regnet_y_32gf(weights=weights)
        model.fc = torch.nn.Linear(3712, head_emb_size)
        
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    elif img_model_name == "regnet_y_128gf":
        if not load_weight:
            model = regnet_y_128gf()
        else:
            weights = RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1
            model = regnet_y_128gf(weights=weights)
        model.fc = torch.nn.Linear(7392, head_emb_size)
        
        preprocess = transforms.Compose([
            transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif img_model_name == "regnet_y_128gf_linear":
        if not load_weight:
            model = regnet_y_128gf()
        else:
            weights = RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_LINEAR_V1
            model = regnet_y_128gf(weights=weights)
        model.fc = torch.nn.Linear(7392, head_emb_size)
        
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
        
    elif img_model_name == "vit_b_16":
        if not load_weight:
            model = vit_b_16(image_size=384)
        else:
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
            model = vit_b_16(weights=weights)
        model.heads.head = torch.nn.Linear(768, head_emb_size)
        
        preprocess = transforms.Compose([
            transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    elif img_model_name == "vit_b_16_linear":
        if not load_weight:
            model = vit_b_16(image_size=224)
        else:
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
            model = vit_b_16(weights=weights)
        model.heads.head = torch.nn.Linear(768, head_emb_size)
        
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    elif img_model_name == "vit_l_16":
        if not load_weight:
            model = vit_l_16(image_size=512)
        else:
            weights = ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
            model = vit_l_16(weights=weights)

        model.heads.head = torch.nn.Linear(1024, head_emb_size)
    
        preprocess = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif img_model_name == "vit_h_14":
        if not load_weight:
            model = vit_h_14(image_size=512)
        else:
            weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
            model = vit_h_14(weights=weights)

        model.heads.head = torch.nn.Linear(1280, head_emb_size)
    
        preprocess = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif img_model_name == "efficientnet_v2_l":
        if not load_weight:
            model = efficientnet_v2_l(image_size=480)
        else:
            weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
            model = efficientnet_v2_l(weights=weights)

        model.classifier[1] = torch.nn.Linear(1280, head_emb_size)
    
        preprocess = transforms.Compose([
            transforms.Resize(480, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(480),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                 std=[0.5, 0.5, 0.5]),
        ])

    elif img_model_name == "efficientnet_v2_m":
        if not load_weight:
            model = efficientnet_v2_m(image_size=480)
        else:
            weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
            model = efficientnet_v2_m(weights=weights)

        model.classifier[1] = torch.nn.Linear(1280, head_emb_size)
    
        preprocess = transforms.Compose([
            transforms.Resize(480, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(480),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif img_model_name == "convnext_large":
        if not load_weight:
            model = convnext_large(image_size=224)
        else:
            weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
            model = convnext_large(weights=weights)

        model.classifier[2] = torch.nn.Linear(1536, head_emb_size)
    
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif img_model_name == "swin_v2_b":
        if not load_weight:
            model = swin_v2_b(image_size=256)
        else:
            weights = Swin_V2_B_Weights.IMAGENET1K_V1
            model = swin_v2_b(weights=weights)

        model.head = torch.nn.Linear(1024, head_emb_size)
    
        preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    return model, preprocess

def create_submission(pred_arr, img_names, text_emb_size):
    imgIds = [i.split('.')[0] for i in img_names]

    EMBEDDING_LENGTH = text_emb_size
    eIds = list(range(EMBEDDING_LENGTH))

    imgId_eId = [
        '_'.join(map(str, i)) for i in zip(
            np.repeat(imgIds, EMBEDDING_LENGTH),
            np.tile(range(EMBEDDING_LENGTH), len(imgIds)))]
    
    submission = pd.DataFrame(
                    index=imgId_eId,
                    data=np.array(pred_arr).flatten(),
                    columns=['val']).rename_axis('imgId_eId')
    return submission

class CustomDataSet(Dataset):
    def __init__(self, data_dir, img2prompt, img_preprocess):
        self.data_dir = data_dir
        self.img_names = list(img2prompt.keys())
        self.img2prompt = img2prompt
        self.img_preprocess = img_preprocess

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        img = Image.open(img_path)
        img_emb = self.img_preprocess(img)
        
        prompt = str(self.img2prompt[img_name])
        
        return img_name, img_emb, prompt
    
######### TRAIN FUNCTIONS #########

def get_train_config(train_size, val_size, batch_size, full_train_epoch_num, full_val_epoch_num):
    max_batches_per_epoch_train = train_size // batch_size // full_train_epoch_num
    max_batches_per_epoch_val = val_size // batch_size // full_val_epoch_num
    return max_batches_per_epoch_train, max_batches_per_epoch_val

def get_loss(loss_name, device):
    if loss_name == "cosine":
        loss_fn = torch.nn.CosineEmbeddingLoss()
        return lambda pred, true: loss_fn(pred, true, torch.ones(pred.size(0)).to(device))
    
def get_scheduler(scheduler_name, overfiting_epoch, lr_start):
    if scheduler_name == "None":
        scheduler = None
    elif scheduler_name == "StepLR":
        scheduler = lambda optimizer: StepLR(
            optimizer, step_size = overfiting_epoch * 0.5, gamma = 0.5
        )
    elif scheduler_name == "ExponentialLR":
        scheduler = lambda optimizer: ExponentialLR(
            optimizer, gamma = 1.05
        )
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = lambda optimizer: CosineAnnealingLR(
            optimizer, T_max=overfiting_epoch, eta_min=lr_start * 0.1
        )
    elif scheduler_name == "CosineAnnealingWarmRestartsLR":
        scheduler = lambda optimizer: CosineAnnealingWarmRestarts(
            optimizer, T_0=overfiting_epoch, T_mult=1, eta_min=lr_start * 0.01
        )
    elif scheduler_name == "CyclicLR":
        scheduler = lambda optimizer: CyclicLR(
            optimizer, 
            base_lr =lr_start / 10, max_lr = lr_start * 10, 
            step_size_up = overfiting_epoch * 0.5, 
            mode = "triangular2", 
            cycle_momentum=False
        )
    return scheduler

def create_summary_writer(log_dir, add_curr_time: bool):
    if add_curr_time:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir += f"_{current_time}"
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def train_epoch(img_model, st_model, train_dataloader, loss_f, optimizer, 
                train_only_head: bool, max_batches_per_epoch_train: int, device, 
                disable_tqdm=True):
    if train_only_head:
        img_model.eval()
        img_model.fc.train()
    else:
        img_model.train()
        
    mean_train_loss = 0
    train_batches_n = 0
    for batch_i, (img_names, img_embs, prompts) in enumerate(tqdm(train_dataloader, disable=disable_tqdm)):
        if batch_i > max_batches_per_epoch_train:
            break
        
        pred = img_model(img_embs.to(device))
        prompts_emb = torch.Tensor(st_model.encode(prompts)).to(device)
        
        loss = loss_f(pred, prompts_emb)

        img_model.zero_grad()
        loss.backward()
        optimizer.step()

        mean_train_loss += float(loss)
        train_batches_n += 1

    mean_train_loss /= train_batches_n
    return mean_train_loss

def val_epoch(img_model, st_model, val_dataloader, loss_f, optimizer, 
              max_batches_per_epoch_val: int, device, test_flip: bool, 
              disable_tqdm=True):
    img_model.eval()
    
    mean_val_loss = 0
    val_batches_n = 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_i, (img_names, img_embs, prompts) in enumerate(tqdm(val_dataloader, disable=disable_tqdm)):
            if batch_i > max_batches_per_epoch_val:
                break
                
            img_embs = img_embs.to(device)
                
            pred = img_model(img_embs)
            
            if test_flip:
                img_embs_flip = transforms.functional.hflip(img_embs)
                pred_flip = img_model(img_embs_flip)
                pred = (pred + pred_flip) / 2
            
            prompts_emb = torch.Tensor(st_model.encode(prompts)).to(device)

            loss = loss_f(pred, prompts_emb)

            mean_val_loss += float(loss)
            val_batches_n += 1
            
    mean_val_loss /= val_batches_n
    return mean_val_loss

######### CFG #########

@dataclass
class CFG_CLASS:
    dataset_dupl_word: int
    img_model_name: str 
    lr_scheduler_name: str
    lr: float
    
    seed: int = 42
    text_emb_size: int = 384
    is_kaggle: bool = (os.environ.get('PWD') == '/kaggle/working')
    train_files_dir: str = "img2emb-data"
    
    save_model: bool = True
    img_model_test_size: float = 0.05

    loss_name: str = "cosine"
    train_only_head: bool = False
    
    train_aug: bool = True
    test_flip: bool = True
    
    full_train_epoch_num: int = 100
    max_epoch_num: int = full_train_epoch_num * 10
    full_val_epoch_num: int = 2
    early_stopping_patience: int = full_val_epoch_num * 25
    
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_name: str = field(init=False)
    metadata_path: str = field(init=False)
    aug_name: str = field(init=False)
    model_name: str = field(init=False)
    
    def __post_init__(self):
        self.dataset_name = f"dataset_duplwords_{self.dataset_dupl_word}"
        self.metadata_path = f"../input/metadata/metadata_duplwords_{self.dataset_dupl_word}.parquet"
        
        self.aug_name: str = f"flip_{int(self.train_aug)}"
        self.model_name = f"model_{self.img_model_name}_sch_{self.lr_scheduler_name}_lr_{self.lr:.0e}".replace("-", "_")
        
        self.train_name = f"{self.dataset_name}_{self.model_name}"
        
        self.batch_size = batch_size_config[self.img_model_name][self.is_kaggle]
        self.num_workers = self.batch_size if not self.is_kaggle else 2

### MAIN ###

if __name__ == "__main__":
    sys.path.append('../input/sentence-transformers-222/sentence-transformers')
    from sentence_transformers import SentenceTransformer, models
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    ### ARGPARSE
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dupl_word', type=int)
    parser.add_argument(
        '--img_model_name', type=str, 
        choices=["vit_b_16", "vit_b_16_linear", "regnet_y_16gf", "regnet_y_32gf", "regnet_y_32gf_linear", "regnet_y_16gf_linear"]
    )
    parser.add_argument('--lr', type=float)
    parser.add_argument(
        '--lr_scheduler_name', type=str, 
        choices=["None", "StepLR", "ExponentialLR", "CosineAnnealingLR", "CosineAnnealingWarmRestartsLR", "CyclicLR"]
    )
    parser.add_argument('--save_model', type=bool, default=True)
    args = parser.parse_args()
    
    ### CFG 
    
    CFG = CFG_CLASS(
        dataset_dupl_word=args.dataset_dupl_word,
        img_model_name=args.img_model_name,
        lr=args.lr,
        lr_scheduler_name=args.lr_scheduler_name,
        save_model=args.save_model,
    )
    set_seed(CFG.seed)
    print("Train name: ", CFG.train_name)
    
    ### Dataset load
    
    train_data_dir = Path("../input/")
    metadata = pd.read_parquet(CFG.metadata_path)
    print("Metadata shape: ", metadata.shape)
    
    full_prompt = metadata[["image_name", "prompt"]].values
    train_prompt, val_prompt = train_test_split(
        full_prompt, 
        test_size=CFG.img_model_test_size, 
        random_state=CFG.seed,
        shuffle=True
    )

    CFG.dataset_train_size = len(train_prompt)
    CFG.dataset_val_size = len(val_prompt)
    CFG.max_batches_per_epoch_train, CFG.max_batches_per_epoch_val = get_train_config(
        train_size=CFG.dataset_train_size, 
        val_size=CFG.dataset_val_size, 
        batch_size=CFG.batch_size, 
        full_train_epoch_num=CFG.full_train_epoch_num, 
        full_val_epoch_num=CFG.full_val_epoch_num
    )

    print("Train/val")
    print("Sizes: ", CFG.dataset_train_size, "/", CFG.dataset_val_size)
    print("Batches per epoch: ", 
          CFG.max_batches_per_epoch_train, "/",
          CFG.max_batches_per_epoch_val)
    print("Images per epoch: ", 
          CFG.max_batches_per_epoch_train * CFG.batch_size, "/",
          CFG.max_batches_per_epoch_val * CFG.batch_size)

    train_prompt_dict = {img_name: prompt for img_name, prompt in train_prompt}
    val_prompt_dict = {img_name: prompt for img_name, prompt in val_prompt}
    
    ### Create model and dataloaders
    
    st_model = SentenceTransformer('../input/sentence-transformers-222/all-MiniLM-L6-v2/')
    img_model, img_preprocess = get_img_model(
        img_model_name=CFG.img_model_name, 
        load_weight=not CFG.is_kaggle, 
        head_emb_size=CFG.text_emb_size
    )
    img_model.to(CFG.device)

    if CFG.train_aug:
        train_img_preprocess = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            img_preprocess,
        ])
    else:
        train_img_preprocess = img_preprocess

    train_dataset = CustomDataSet(
        data_dir=train_data_dir, 
        img2prompt=train_prompt_dict, 
        img_preprocess=train_img_preprocess,
    )
    val_dataset = CustomDataSet(
        data_dir=train_data_dir, 
        img2prompt=val_prompt_dict, 
        img_preprocess=img_preprocess,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True,
                                        num_workers=CFG.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=True,
                                      num_workers=CFG.num_workers)
    
    ### Train model
    
    writer = create_summary_writer(log_dir=f"../logs/{CFG.dataset_name}/{CFG.model_name}", add_curr_time=True)

    loss_f = get_loss(loss_name=CFG.loss_name, device=CFG.device)
    optimizer = torch.optim.Adam(img_model.parameters(), lr=CFG.lr)

    lr_scheduler = get_scheduler(
        scheduler_name=CFG.lr_scheduler_name, 
        overfiting_epoch=CFG.full_train_epoch_num, 
        lr_start=CFG.lr
    )

    if lr_scheduler:
        lr_scheduler = lr_scheduler(optimizer)

    img_model.to(CFG.device)
    st_model.to(CFG.device)

    best_val_loss = float('inf')
    best_epoch_i = 0

    for epoch_i in range(CFG.max_epoch_num):
        mean_train_loss = train_epoch(img_model, st_model, train_dataloader, loss_f, optimizer, 
                                      disable_tqdm=True, train_only_head=CFG.train_only_head, 
                                      max_batches_per_epoch_train=CFG.max_batches_per_epoch_train, 
                                      device=CFG.device)
        mean_val_loss = val_epoch(img_model, st_model, val_dataloader, loss_f, optimizer, disable_tqdm=True, 
                                  max_batches_per_epoch_val=CFG.max_batches_per_epoch_val, device=CFG.device, 
                                  test_flip=CFG.test_flip)

        if lr_scheduler:
            lr_scheduler.step()

        train_sim = round(1 - mean_train_loss, 3)
        val_sim = round(1 - mean_val_loss, 3)

        ### SAVE BEST MODEL ###
        print(f"Epoch {epoch_i + 1}, lr={optimizer.param_groups[0]['lr']:.2e}: train = {train_sim}, val = {val_sim}", end="; ")

        if mean_val_loss < best_val_loss:
            best_epoch_i = epoch_i
            best_val_loss = mean_val_loss

            if CFG.save_model:
                torch.save(
                    img_model.state_dict(), f"../input/{CFG.train_files_dir}/{CFG.train_name}.torch"
                )
            print(f'new best model')
        elif epoch_i - best_epoch_i > CFG.early_stopping_patience:
            print(f'early stopping')
            break
        else:
            print("continue")

        ### HISTORY ###
        writer.add_scalars(
            "Similarity",
            {"train": 1 - mean_train_loss, "val": 1 - mean_val_loss}, 
            global_step=epoch_i + 1
        )
        writer.flush()
