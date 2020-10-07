import torch,numpy as np,cv2
from torch.autograd import Variable
from PIL import Image
from  matplotlib import pyplot as plt
from  skimage.measure import compare_ssim,compare_psnr


'''
def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)

def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)

def psnr(img1, img2):
   import math
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 1003

   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
'''



device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

model=torch.load('E:\\practice\\DRD\\14model.pt')
# model=torch.load('C:\\Users\\ASUS\\Desktop\\modelfile\\7model.pt')
model.to(device)



if __name__ == '__main__':
    model.eval()
    num=input("num:")
    # img1 = cv2.imread('E:\\practice\\GCAnet\\rain\\train\\%s.png' % num)
    img1  = cv2.imread('C:\\Users\\ASUS\\Desktop\\rain\\%s.jpg' % num)
    image = np.array(Image.open('E:\\practice\\GCAnet\\rain\\label\\%s.png' % num))
    blur = cv2.medianBlur(img1, 7)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    img = torch.Tensor(img1.transpose((2, 0, 1)))#-torch.Tensor(blur.transpose((2, 0, 1)))
    img = torch.unsqueeze(img, dim=0) / 255.0
    img = Variable(img.to(device))
    with torch.no_grad():
        rain_edge1,clear_edge1=model(img)
    out=rain_edge1
    # edge=(edge_compute(out).data[0].cpu())*255
    out_img_data = (out.data[0].cpu()*255).round().clamp(0, 255)
    out_img = out_img_data.numpy().astype(np.uint8).transpose(1, 2, 0)
    # number=0
    # i=1
    # for n,value in enumerate(rain.ravel()):
    #     if value!=0:
    #         number=number+1
    #     i=n+1
    # print(number/i)
    gray2=cv2.cvtColor(out_img, cv2.COLOR_RGB2GRAY)

    edge_x=cv2.Sobel(gray2,cv2.CV_8U,1,0)
    edge_y = cv2.Sobel(gray2, cv2.CV_8U,0,1)

    # ret,edge_x=cv2.threshold(edge_x,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    a=np.zeros_like(gray2)
    for i in range(gray2.shape[0]):
        for j in range(gray2.shape[1]):
            if edge_y[i][j]<10 and edge_x[i][j]>80:
                a[i][j] = edge_x[i][j]
            else:
                a[i][j] = 0
    a=a.astype('uint8')
    kerenl3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))  #1,3,2
    a = cv2.erode(a, kerenl3, iterations=2)
    # kerenl1=cv2.getStructuringElement(cv2.MORPH_RECT,(2,3))  #2,3,1
    # a=cv2.dilate(a,kerenl1,iterations=1)
    # kerenl2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    # a=cv2.erode(a,kerenl2,iterations=1)
    cv2.imshow('edge_y',a)
    cv2.imshow('edge_x', edge_x)


    cv2.waitKey()

