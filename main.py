from common import *
from networks import *
import torch  
import torch.nn as nn
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary

def main():
    Nx = 1000
    Ny = 1000
    z = 857
    wavelength = 0.635
    deltaX = 1.67
    deltaY = 1.67
    
    img = Image.open('./Image1.bmp')

    # pytorch provides a function to convert PIL images to tensors.
    pil2tensor = transforms.ToTensor()
    tensor2pil = transforms.ToPILImage()
    
    tensor_img = pil2tensor(img)
    
    g = tensor_img.numpy()
    g = np.sqrt(g)
    g = (g-np.min(g))/(np.max(g)-np.min(g))
    
    plt.figure(figsize=(20,15))
    plt.imshow(np.squeeze(g), cmap='gray')
    
    phase = propagator(Nx,Ny,z,wavelength,deltaX,deltaY)
    eta = np.fft.ifft2(np.fft.fft2(g)*np.fft.fftshift(np.conj(phase)))
    plt.figure(figsize=(20,15))
    plt.imshow(np.squeeze(np.abs(eta)), cmap='gray')
    
    
    criterion_1 = RECLoss()
    model = Net().cuda()
    optimer_1 = optim.Adam(model.parameters(), lr=5e-3)
    
    
    device = torch.device("cuda")
    epoch_1 = 5000
    epoch_2 = 2000
    period = 100
    eta = torch.from_numpy(np.concatenate([np.real(eta)[np.newaxis,:,:], np.imag(eta)[np.newaxis,:,:]], axis = 1))
    holo = torch.from_numpy(np.concatenate([np.real(g)[np.newaxis,:,:], np.imag(g)[np.newaxis,:,:]], axis = 1))
    
    for i in range(epoch_1):
        in_img = eta.to(device)
        target = holo.to(device)
        
        out = model(in_img) 
        l1_loss = criterion_1(out, target)
        loss = l1_loss
        
        
        optimer_1.zero_grad()
        loss.backward()
        optimer_1.step()
        
        print('epoch [{}/{}]     Loss: {}'.format(i+1, epoch_1, l1_loss.cpu().data.numpy()))
        if ((i+1) % period) == 0:
            outtemp = out.cpu().data.squeeze(0).squeeze(1)
            outtemp = outtemp
            plotout = torch.sqrt(outtemp[0,:,:]**2 + outtemp[1,:,:]**2)
            plotout = (plotout - torch.min(plotout))/(torch.max(plotout)-torch.min(plotout))
            plt.figure(figsize=(20,15))
            plt.imshow(tensor2pil(plotout), cmap='gray')
            plt.show()
            
            plotout_p = (torch.atan(outtemp[1,:,:]/outtemp[0,:,:])).numpy()
            plotout_p = Phase_unwrapping(plotout_p)
            plotout_p = (plotout_p - np.min(plotout_p))/(np.max(plotout_p)-np.min(plotout_p))
            plt.figure(figsize=(20,15))
            plt.imshow((plotout_p), cmap='gray')
            plt.show()
    
    outtemp = out.cpu().data.squeeze(0).squeeze(1)
    outtemp = outtemp
    plotout = torch.sqrt(outtemp[0,:,:]**2 + outtemp[1,:,:]**2)
    plotout = (plotout - torch.min(plotout))/(torch.max(plotout)-torch.min(plotout))
    plt.figure(figsize=(30,30))
    plt.imshow(tensor2pil(plotout), cmap='gray')
    plt.show()
    
    
    plotout_p = (torch.atan(outtemp[1,:,:]/outtemp[0,:,:])).numpy()
    plotout_p = Phase_unwrapping(plotout_p)
    plotout_p = (plotout_p - np.min(plotout_p))/(np.max(plotout_p)-np.min(plotout_p))
    plt.figure(figsize=(30,30))
    plt.imshow((plotout_p), cmap='gray')
    plt.show()
if __name__ == '__main__':
    main()