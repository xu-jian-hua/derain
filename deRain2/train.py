import torch,torch.nn as nn
from  torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
from tensorboardX import SummaryWriter
from trans import datas

from trans import netmodel

sw=SummaryWriter(log_dir='E:\\practice\\GCAnet\\log')


t=transforms.Compose([transforms.CenterCrop(256),transforms.ToTensor()])

dataset=datas.Data('E:\\practice\\GCAnet\\picture\\train','E:\\practice\\GCAnet\\picture\\label',t)
data=DataLoader(dataset=dataset,batch_size=6,shuffle=True)

device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
model = netmodel.Net().to(device)





opt=optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.001)
loss_func=nn.MSELoss().to(device)


if __name__ == '__main__':
    model.train(True)
    for epoch in range(20):
        print("===epoc===%d"%epoch)
        for i,(img,target) in enumerate(data):
            img = Variable(img.to(device))
            target = Variable(target.to(device))
            torch.cuda.empty_cache()
            rain,out=model(img)
            # re=0
            # for param in model.parameters():
            #     re=re+torch.sum(torch.abs(param))
            loss=loss_func(out.clamp(0, 1),target)+loss_func((rain+out).clamp(0, 1),img)
            # loss = loss_func(out.clamp(0, 1), target)
            print(loss.item())

            opt.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)
            opt.step()
            if i %200==0 :
                # sw.add_scalar("loss",loss.item(),epoch*1060+i)
                torch.save(model,'E:\\practice\\trans\\modelfile\\{0}_{1}model.pt'.format(epoch,i))
        # for p in opt.param_groups:
        #     p['lr'] *= 0.1
