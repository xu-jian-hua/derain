import torch,numpy as np,cv2
from torch.autograd import Variable
from PIL import Image
from  matplotlib import pyplot as plt
from torchvision import transforms
#from GCAnet import DerainAndDefog
# from GCAnet import frNet
# from  GCAnet import net
def psnr(img1, img2):
   import math
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 1003

   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


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


device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

# model=torch.load('E:\\practice\\trans\\15_0model.pt')
model=torch.load('E:\\practice\\trans\\modelfile\\2_0model.pt')
model.to(device)



if __name__ == '__main__':
    model.eval()
    num=input("num:")
    # img1 = np.array(Image.open('C:\\Users\\ASUS\\Desktop\\%s.png' % num))

    img1 = np.array(Image.open('E:\\practice\\GCAnet\\picture\\train\\%s.jpg' % num))
    image = np.array(Image.open('E:\\practice\\GCAnet\\picture\\label\\%s.jpg' % num))
    # img1 = np.array(Image.open('F:\\picture\\weather\\weather\\rain\\RainTrainH\\RainTrainH\\rain-%s.png' % num))
    # image = np.array(Image.open('F:\\picture\\weather\\weather\\rain\\RainTrainH\\RainTrainH\\norain-%s.png' % num))
    w,h,c=img1.shape
    img1, image = img1[0:(w // 4 * 4), 0:(h // 4 * 4)],image[0:(w // 4 * 4), 0:(h // 4 * 4)]


    # image= np.array(Image.open('F:\\picture\\weather\\weather\\rain\\Rain12600\\Rain12600\\ground_truth\\%s.jpg' % num))
    # img= np.array(Image.open('F:\\picture\\weather\\weather\\rain\\Rain12600\\Rain12600\\rainy_image\\%s_5.jpg' % num))
    # img = np.array(Image.open('E:\\GCANet-master\\examples\\%s.png' % num))
    # img = np.array(Image.open('E:\\practice\\GCAnet\\test\\%s.jpg' % num))
    # img =np.array(Image.open('E:\\practice\\GCAnet\\p\\train\\%s.jpg'%num))
    #
    # image = np.array(Image.open('E:\\practice\\GCAnet\\p\\label\\%s.jpg'%num))
    img = torch.Tensor(img1.transpose((2, 0, 1)))
    img = torch.unsqueeze(img, dim=0) / 255.0
    # tensor=edge_compute(img)
    # img = torch.unsqueeze(torch.cat((img,tensor),dim=0),dim=0)/255.0
    img = Variable(img.to(device))
    with torch.no_grad():
        rain,out1=model(img)
        rain, out2 = model(out1)
        # rain, out3 = model(out2)
    out=out1
    out_img_data = (out.data[0].cpu()*255).round().clamp(0, 255)
    out_img = out_img_data.numpy().astype(np.uint8).transpose(1, 2, 0)
    rain = (img.data[0].cpu() * 255-out.data[0].cpu() * 255).round().clamp(0, 255)
    # rain = (rain.data[0].cpu() * 255).round().clamp(0, 255)
    rain = rain.numpy().astype(np.uint8).transpose(1, 2, 0)
    # out_img=cv2.blur(out_img,(3,3))
    # print(psnr(img1,image))
    # print(psnr(out_img, image))
    e=Image.fromarray(out_img)
    plt.figure(figsize=(10,10))
    plt.suptitle("psnr: {0}-->{1}".format(round(psnr(img1, image), 2), round(psnr(out_img, image), 2)), fontsize=18,
                 y=0.05)
    plt.subplot(2,2,1)
    plt.imshow(image)
    plt.subplot(2,2,2)
    plt.imshow(img1)
    plt.subplot(2,2,3)
    plt.imshow(out_img)
    plt.subplot(2,2, 4)
    plt.imshow(rain)
    plt.show()