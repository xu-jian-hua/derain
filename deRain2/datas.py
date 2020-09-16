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
    # edge=edge_compute(img)
    # image=torch.cat((img,edge),dim=0)
    return img,target,


#用于GCANet
def edge_compute(x):
  x_diffx = torch.abs(x[:, :, 1:] - x[:, :, :-1])
  x_diffy = torch.abs(x[:, 1:, :] - x[:, :-1, :])
  y = torch.zeros_like(x)
  y[:, :, 1:] += x_diffx
  y[:, :, :-1] += x_diffx
  y[:, 1:, :] += x_diffy
  y[:, :-1, :] += x_diffy
  # y = torch.sum(y, 0, keepdim=True) / 3

  y /= 4
  return y