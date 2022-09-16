import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class MyImageNet(Dataset):

    def __init__(self,
                 ann_path,
                 img_dir,
                 transforms):
        self.ann_path = ann_path
        self.img_dir = img_dir
        with open(ann_path, 'r') as f:
            lines = f.read().splitlines()
        img_path_list = []
        img_label_list = []
        for line in lines:
            sp = line.split(' ')
            img_path_list.append(os.path.join(img_dir, sp[0]))
            img_label_list.append(int(sp[1]))
        self.img_label_list = img_label_list
        self.img_path_list = img_path_list
        self.transforms = transforms

    def __len__(self):
        return len(self.img_label_list)

    def __getitem__(self, idx):
        label = self.img_label_list[idx]
        try:
            img = Image.open(self.img_path_list[idx])
        except:
            img = Image.open(self.img_path_list[1]) 
        img = img.convert('RGB')
        img_tensor = self.transforms(img)
        label_tensor = torch.squeeze(torch.LongTensor([label]))
        return img_tensor, label_tensor
