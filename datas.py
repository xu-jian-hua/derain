from torch.utils.data import Dataset
from  torchvision import transforms
import numpy as np,os,torch
from PIL import Image

class Data(Dataset):
  def __init__(self, path_img, path_target, transforms=None):
    self.path_img=path_img
    self.path_target=path_target
    self.train = os.listdir(path_img)
    self.train.sort(key=lambda x:int(x.split('.')[0]))
    self.targets = os.listdir(path_target)
    self.targets.sort(key=lambda  x:int(x.split('.')[0]))
    self.transforms = transforms


  def __len__(self):
    return len(self.train)

  def __getitem__(self, idx):
    img = Image.open(os.path.join(self.path_img,self.train[idx]))
    target = Image.open(os.path.join(self.path_target,self.targets[idx]))
    img=self.transforms(img)
    target=self.transforms(target)
    return img,target,


