from torch.autograd import Variable
import torch,torch.nn as nn,time
from  torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
from DRD.datas import Data
from DRD.net import Net
from math import exp
import torch.nn.functional as F

# sw=SummaryWriter(log_dir='E:\\practice\\DRD\\log')



t=transforms.Compose([transforms.CenterCrop(256),transforms.ToTensor()])

dataset=Data('E:\\practice\\GCAnet\\picture\\train','E:\\practice\\GCAnet\\picture\\label',t)
data = DataLoader(dataset=dataset,batch_size=4,shuffle=True)

device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
model = Net().to(device)


opt=optim.Adam(model.parameters(),lr=0.001)#
loss_func = nn.MSELoss().to(device)



if __name__ == '__main__':
    model.train(True)
    for epoch in range(15):
        since=time.time()
        print("===epoc===%d"%epoch)
        for i,(img,target,edge,blur) in enumerate(data):
            img = img.to(device)
            target = target.to(device)
            edge = edge.to(device)
            blur = blur.to(device)
            rain_edge,img_edge=model(img)

            loss = loss_func(img_edge, target) + loss_func((rain_edge + img_edge), img)
            # loss=loss_func(img_edge,target)+loss_func(rain_edge,img-target)
            opt.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)
            opt.step()
            # schedular.step()
            if i %2==0 :
                print("loss:", loss.item())
                # sw.add_scalar("loss",loss.item(),epoch*1060+i)
        print("epoch cost time:", time.time() - since)
        torch.save(model,'E:\\practice\\DRD\\modelfile\\{0}model.pt'.format(epoch))