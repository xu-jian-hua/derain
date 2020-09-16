import torch
import torch.nn as nn
import torch.nn.functional as F

class DeRain2(nn.Module):
    def __init__(self,numList):
        # [3,16,32,3]
        super(DeRain2,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(numList[0], numList[1], 3, 1, 1),
                                   nn.BatchNorm2d(numList[1]),nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(numList[1], numList[1], 3, 1, 1),
                                    nn.BatchNorm2d(numList[1]),nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(numList[1], numList[1], 3, 2, 1),
                                    nn.BatchNorm2d(numList[1]),nn.ReLU(True))

        self.conv1_1 = nn.Sequential(nn.Conv2d(numList[1],numList[1], 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm2d(numList[1]), nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(numList[1],numList[1], 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm2d(numList[1]), nn.ReLU(True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(numList[1],numList[1], 5, 1, padding=2, dilation=1),
                                     nn.BatchNorm2d(numList[1]), nn.ReLU(True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(numList[1],numList[1], 5, 1, padding=2, dilation=1),
                                     nn.BatchNorm2d(numList[1]), nn.ReLU(True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(numList[1],numList[1], 7, 1, padding=3, dilation=1),
                                     nn.BatchNorm2d(numList[1]), nn.ReLU(True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(numList[1],numList[1], 7, 1, padding=3, dilation=1),
                                     nn.BatchNorm2d(numList[1]), nn.ReLU(True))
        self.gate1=nn.Conv2d(numList[1]*3,3,3,1,1)

        self.conv4 = nn.Sequential(nn.Conv2d(numList[1], numList[2], 3, 1, 1),
                                   nn.BatchNorm2d(numList[2]), nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(numList[2], numList[2], 3, 1, 1),
                                   nn.BatchNorm2d(numList[2]), nn.ReLU(True))
        self.conv6 = nn.Sequential(nn.Conv2d(numList[2], numList[2], 3, 2, 1),
                                   nn.BatchNorm2d(numList[2]), nn.ReLU(True))

        self.conv4_1 = nn.Sequential(nn.Conv2d(numList[2], numList[2], 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm2d(numList[2]), nn.ReLU(True))
        self.conv4_2 = nn.Sequential(nn.Conv2d(numList[2], numList[2], 3, 1, padding=1, dilation=1),
                                     nn.BatchNorm2d(numList[2]), nn.ReLU(True))
        self.conv5_1 = nn.Sequential(nn.Conv2d(numList[2], numList[2], 5, 1, padding=2, dilation=1),
                                     nn.BatchNorm2d(numList[2]), nn.ReLU(True))
        self.conv5_2 = nn.Sequential(nn.Conv2d(numList[2], numList[2], 5, 1, padding=2, dilation=1),
                                     nn.BatchNorm2d(numList[2]), nn.ReLU(True))
        self.conv6_1 = nn.Sequential(nn.Conv2d(numList[2], numList[2], 7, 1, padding=3, dilation=1),
                                     nn.BatchNorm2d(numList[2]), nn.ReLU(True))
        self.conv6_2 = nn.Sequential(nn.Conv2d(numList[2], numList[2], 7, 1, padding=3, dilation=1),
                                     nn.BatchNorm2d(numList[2]), nn.ReLU(True))

        self.gate2 = nn.Conv2d(numList[2] * 3, 3, 3, 1, 1)
        self.deconv6 = nn.Sequential(nn.ConvTranspose2d(numList[2]*2, numList[2], 4, 2, 1),
                                    nn.BatchNorm2d(numList[2]), nn.ReLU(True))
        self.deconv5 = nn.Sequential(nn.Conv2d(numList[2] , numList[2], 3, 1, 1),
                                     nn.BatchNorm2d(numList[2]), nn.ReLU(True))
        self.deconv4 = nn.Sequential(nn.Conv2d(numList[2] , numList[2], 3, 1, 1),
                                     nn.BatchNorm2d(numList[2]), nn.ReLU(True))

        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(numList[1] * 4, numList[1], 4, 2, 1),
                                     nn.BatchNorm2d(numList[1]), nn.ReLU(True))
        self.deconv2 = nn.Sequential(nn.Conv2d(numList[1], numList[1], 3, 1, 1),
                                     nn.BatchNorm2d(numList[1]), nn.ReLU(True))
        self.deconv1 = nn.Sequential(nn.Conv2d(numList[1], numList[0], 3, 1, 1),
                                     nn.BatchNorm2d(numList[0]), nn.ReLU(True))




    def forward(self, x):
        a1 = self.conv1(x)
        a2=self.conv2(a1)
        y1=self.conv3(a2)
        u1=self.conv1_2(self.conv1_1(y1))
        u2 = self.conv2_2(self.conv2_1(y1))
        u3=self.conv3_2(self.conv3_1(y1))
        gate1=self.gate1(torch.cat((u1,u2,u3),dim=1))
        b1 = self.conv5(self.conv4(y1))
        y2 = self.conv6(b1)
        v1 = self.conv4_2(self.conv4_1(y2))
        v2 = self.conv5_2(self.conv5_1(y2))
        v3 = self.conv6_2(self.conv6_1(y2))
        gate2 = self.gate2(torch.cat((v1, v2, v3), dim=1))
        y3=self.deconv6(torch.cat((y2,v1*gate2[:,[0],:,:]+v2*gate2[:,[1],:,:]+v3*gate2[:,[2],:,:]),dim=1))
        b2=self.deconv4(self.deconv5(y3))
        y4 = self.deconv3(torch.cat((y1, u1*gate1[:,[0],:,:] + u2*gate1[:,[1],:,:] + u3*gate1[:,[2],:,:],b2), dim=1))
        a3 = self.deconv2(y4)
        a4=self.deconv1(a3)



        return a4



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        numList=[3,24,48]
        self.rain=DeRain2(numList=numList)


    def forward(self, x):
        a1=self.rain(x)

        x1=x-a1
        return  (a1,x1)
