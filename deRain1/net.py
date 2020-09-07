import torch
import torch.nn as nn
import torch.nn.functional as F


class DeRain(nn.Module):
    def __init__(self,inNum=3,outNum=3):
        super(DeRain,self).__init__()
        self.conv1_1=nn.Sequential(nn.Conv2d(inNum,8,3,1,padding=1,dilation=1),nn.BatchNorm2d(8),nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(8, 32, 3, 1, padding=1, dilation=1), nn.BatchNorm2d(32),nn.ReLU(True))

        self.conv2_1=nn.Sequential(nn.Conv2d(inNum,8,5,1,padding=2,dilation=1),nn.BatchNorm2d(8),nn.ReLU(True))
        self.conv2_2 = nn.Sequential( nn.Conv2d(8, 32, 5, 1, padding=2, dilation=1),nn.BatchNorm2d(32), nn.ReLU(True))

        self.conv3_1 = nn.Sequential(nn.Conv2d(inNum, 8, 7, 1, padding=3, dilation=1), nn.BatchNorm2d(8), nn.ReLU(True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(8, 32, 7, 1, padding=3, dilation=1),nn.BatchNorm2d(32), nn.ReLU(True))

        self.conv4=nn.Sequential(nn.Conv2d(35, 64, 3, 1, 1, 1),nn.BatchNorm2d(64), nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(99, 128, 3, 1, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.conv6 = nn.Sequential(nn.Conv2d(99+128, outNum, 3, 1, 1, 1), nn.BatchNorm2d(3), nn.ReLU(True))

    def forward(self,x):
        # layer1=torch.cat((x,self.conv1_2(self.conv1_1(x)),self.conv2_2(self.conv2_1(x)),self.conv3_2(self.conv3_1(x))),dim=1)
        layer1 = self.conv1_2(self.conv1_1(x))+ self.conv2_2(self.conv2_1(x))+ self.conv3_2(self.conv3_1(x))
        layer1=torch.cat((x,layer1),dim=1)  #11

        layer2=torch.cat((layer1,self.conv4(layer1)),dim=1)  #43
        layer3=torch.cat((layer2,self.conv5(layer2)),dim=1)#107
        layer4=self.conv6(layer3)
        return  layer4


class DeFog(nn.Module):
    def __init__(self,inNum=3,outNum=3):
        super(DeFog,self).__init__()
        self.conv1 = nn.Conv2d(inNum, 16, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, 2, 1)
        self.norm3 = nn.BatchNorm2d(32)

        self.res1 = Res(32)
        self.res2 = Res(32)

        self.gate = nn.Conv2d(32 * 3, 3, 3, 1, 1)

        self.deconv3 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.norm4 = nn.BatchNorm2d(32)
        self.deconv2 = nn.Conv2d(64, 16, 3, 1, 1)
        self.norm5 = nn.BatchNorm2d(16)
        self.deconv1 = nn.Conv2d(16, outNum, 1, 1)
    def forward(self,x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = F.relu(self.norm2(self.conv2(y)))
        y1 = F.relu(self.norm3(self.conv3(y)))
        y2=self.res1(y1)
        y3=self.res2(y2)

        gates = self.gate(torch.cat((y1, y2, y3), dim=1))
        gated_y = y1 * gates[:, [0], :, :] + y2 * gates[:, [1], :, :] + y3 * gates[:, [2], :, :]
        y = torch.cat((y,F.relu(self.norm4(self.deconv3(gated_y)))),dim=1)
        y = F.relu(self.norm5(self.deconv2(y)))
        y = self.deconv1(y)
        return y

class Res(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(Res, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group),
            nn.BatchNorm2d(channel_num), nn.ReLU(True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel_num, channel_num // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channel_num // 8, channel_num),
            nn.Sigmoid()
        )

    def forward(self, input):
        b, c, _, _ = input.size()
        x = self.conv1(input)
        x = self.conv2(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return (x * y.expand_as(input) + input)

class Net(nn.Module):
    def __init__(self, inNum=3, outNum=3):
        super(Net, self).__init__()
        self.rain1=DeRain(inNum, outNum)
        self.fog1=DeFog(inNum, outNum)
        self.conv1=nn.Sequential(nn.Conv2d(6,2,3,1,1),nn.BatchNorm2d(2),nn.ReLU(True))

    def forward(self, x):
        y1=self.rain1(x)
        z1=self.fog1(x)
        mid=torch.cat((y1,z1),dim=1)
        mid=self.conv1(mid)
        a=y1*mid[:,[0],:,:]+z1*mid[:,[1],:,:]
        x1=x-a
        return  (a,x1)
