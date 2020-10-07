from torch.utils.data import Dataset
from  torchvision import transforms
import numpy as np,os,torch,cv2
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
    img = cv2.imread(os.path.join(self.path_img, self.train[idx]))
    target = cv2.imread(os.path.join(self.path_target, self.targets[idx]))
    blur=cv2.medianBlur(img,7)
    img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    target= Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
    blur = Image.fromarray(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
    img=self.transforms(img)
    target=self.transforms(target)
    blur = self.transforms(blur)
    return img,target,img-blur,blur
