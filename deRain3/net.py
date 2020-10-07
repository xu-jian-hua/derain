import torch.nn as nn, torch


class Mul(nn.Module):
    def __init__(self,num=16):
        super(Mul,self).__init__()
        self.num=num
        self.pad1 = nn.ReplicationPad2d([3, 3, 2, 2])
        self.pad2 = nn.ReplicationPad2d([2, 2, 3, 3])
        self.conv1=nn.Sequential(nn.Conv2d(self.num,self.num,[5,7],1,bias=False),nn.BatchNorm2d(self.num),nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.num, self.num, [7, 5], 1 , bias=False),nn.BatchNorm2d(self.num),nn.ReLU())
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Linear(self.num*2, self.num*2 // 8, bias=False), nn.ReLU(),
                                nn.Linear(self.num*2 // 8, self.num*2, bias=False), nn.Sigmoid())
    def forward(self,x):

        conv1=self.conv1(self.pad1(x))
        conv2 = self.conv2(self.pad2(x))
        concat=torch.cat((conv1,conv2),dim=1)
        b, c, _, _ = concat.size()
        pool=self.pool(concat)
        gates=self.se(pool.view(b,c)).view(b, c, 1, 1)
        gates=gates.expand_as(concat)
        return x+gates[:,0:self.num,:,:]*conv1+gates[:,self.num:self.num*2,:,:]*conv2


class Net(nn.Module):
    def __init__(self,num=16):
        super(Net,self).__init__()
        self.num=num
        self.pad1 = nn.ReplicationPad2d([1, 1, 1, 1])
        self.conv1 = nn.Sequential(nn.Conv2d(3, self.num, 3, 1, bias=False), nn.BatchNorm2d(self.num),nn.ReLU())
        self.mul1=Mul(self.num)
        self.pool1=nn.AvgPool2d(2,2)
        self.mul2 = Mul(self.num)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.mul3 = Mul(self.num)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.mul4=Mul(self.num)
        self.pool4=nn.AvgPool2d(2,2)
        self.mul5=Mul(self.num)
        self.pool5=nn.AvgPool2d(2,2)
        self.mul6=Mul(self.num)
        self.pool6=nn.AvgPool2d(2,2)
        self.mul7 = Mul(self.num)
        self.pool7 = nn.AvgPool2d(2, 2)
        self.mul8 = Mul(self.num)
        self.pool8 = nn.AvgPool2d(2, 2)
        self.convert=nn.Sequential(nn.ConvTranspose2d(self.num*8,self.num,4,2,1,bias=False),nn.BatchNorm2d(self.num),nn.ReLU())
        self.pad2 = nn.ReplicationPad2d([1, 1, 1, 1])
        self.conv2=nn.Sequential(nn.Conv2d(self.num,3,3,1,bias=False),nn.BatchNorm2d(3),nn.ReLU())





    def forward(self,x):

        conv1=self.conv1(self.pad1(x))
        muls=self.mul1(conv1)
        muld=muls
        pool1=self.pool1(muls)
        muls=self.mul2(muld)+muld
        muld = muls
        pool2=self.pool2(muls)
        muls = self.mul3(muld) + muld
        muld = muls
        pool3 = self.pool3(muls)
        muls = self.mul4(muld) + muld
        muld = muls
        pool4 = self.pool4(muls)
        muls = self.mul5(muld) + muld
        muld = muls
        pool5 = self.pool5(muls)
        muls = self.mul6(muld) + muld
        muld = muls
        pool6 = self.pool6(muls)
        muls=self.mul7(muld)+muld
        muld = muls
        pool7=self.pool7(muls)
        muls = self.mul8(muld) + muld
        muld = muls
        pool8 = self.pool8(muls)
        conv2=self.conv2(self.pad2(self.convert(torch.cat((pool1,pool2,pool3,pool4,pool5,pool6,pool7,pool8),dim=1))+muld+conv1))
        return conv2,x-conv2










