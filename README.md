# DeepDIH
The repo shows the corresponding codes of the paper: 
[Deep DIH : Statistically Inferred Reconstruction of Digital In-Line Holography by Deep Learning](https://arxiv.org/abs/2004.12231)

In this paper, we propose a novel DL method that takes advantages of the main characteristic of auto-encoders for blind single-shot hologram reconstruction solely based on the captured sample and without the need for a large dataset of samples with available ground truth to train the model. The simulation results demonstrate the superior performance of the proposed method compared to the state-of-the-art methods used for single-shot hologram reconstruction.

If you have any question, please contact the author: hl459@nau.edu
## File list:
You can also review the existed rusults on .html file or .ipynb file 
- **Complex_conv.html**:
- **DeepDIH.html** / **DeepDIH.ipynb** 
- **main.py**

**Noting**:The HTML and Notebook could be also found in https://drive.google.com/drive/folders/13o86AYWUPvxQxanq4cHxIiDjW22vOd75?usp=sharing
## Requirement:
- GPU memory > 8 GB
- Python 3
- PyTorch(=1.6.0) install:

`conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`(anaconda)

`pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html`

- OpenCV for Python install:

`pip install opencv-contrib-python`

- torchsummary
`pip install torchsummary`

- [git large file storage](https://git-lfs.github.com/)

For more information, check:
- https://pytorch.org/
- https://pypi.org/project/opencv-python/
- https://pypi.org/project/torchsummary/


## Installation
- Clone this repository.
`git lfs clone https://github.com/XiwenChen-NAU/DeepDIH.git`
- `cd DeepDIH`
- run
`python main.py`
- The ouputs (amplitude and phase) in the subfolder `./results`
## Optical paras (pre-defined in `main.py`):
- Spherical light function Nx, Ny :`Nx = 1000 Ny = 1000` 
- hologram size z:`z = 857`
- object-sensor distance wavelength:`wavelength = 0.635`
- wavelength of light deltaX, deltaY : `deltaX = 1.67 deltaY = 1.67`
- If you want to setup your paras, go `main.py` and modify them in:
`main(Nx = *, Ny = *, z = *, wavelength = *, deltaX = *, deltaY = *)`
then run it.
## Network Structure
The objective function can be formulated as:

![image](https://latex.codecogs.com/gif.latex?w%20=%20\mathop{\arg\min}_{w}%20\%20\%20\|%20H-T(f(H,w))\|_{2}^{2})

where we want to propagate the reconstructed object wave to the hologram plane with transmission $T$ and minimize the error between the captured hologram and the forward-propagated result.
![image](https://github.com/XiwenChen-NAU/DeepDIH/blob/master/Figures/fig4-2.jpg)
Deep convolutional autoencoder with “hourglass” architecture. Batch normalization is deployed after each convolution layer except for the last three layers to stabilize the training steps. The hyper-parameters (e.g., the kernel size and feature channels for each layer) is shown. The network is fully convolutional that enables us to feed inputs with different sizes.
## Our Experiments
We implement our model using the PyTorch Framework in a GPU workstation with an NVIDIA Quadro RTX5000 graphics card. Adam optimizer is adopted with a fixed learning rate of 0.0005 for simulation-based experiments and 0.01 for optical experiments. We train the network with an angular spectrum propagation (ASP) back-propagation reconstruction as input for 1500 to 3500 iterations for simulated holograms, and 2500 to 5000 iterations for real-world holograms, respectively.
## Results
![image](https://github.com/XiwenChen-NAU/DeepDIH/blob/master/Figures/fig12.jpg)
Optical Experimental hologram of USAF Resolution Chart and reconstructions. (A) The captured hologram. (B) Amplitude reconstruction with our method. (C) The reconstructed quantitative phase with our method.


## Update loss function compatible for torch >1.6 (torch.fft has been substantially updated in nee version)
```
class RECLoss(nn.Module):
    def __init__(self):
        super(RECLoss,self).__init__()
        self.Nx = 500
        self.Ny = 500
        
        self.wavelength = wavelength
        self.deltaX = deltaX
        self.deltaY = deltaY
        # self.z = z
        # self.prop = self.propagator(self.Nx,self.Ny,self.z,self.wavelength,self.deltaX,self.deltaY)
        # self.prop = self.prop.cuda()

    def propagator(self,Nx,Ny,z,wavelength,deltaX,deltaY):
        k = 1/wavelength
        # x = np.expand_dims(np.arange(np.ceil(-Nx/2),np.ceil(Nx/2),1)*(1/(Nx*deltaX)),axis=0)
        x =torch.unsqueeze(torch.arange(\
                                        torch.ceil(-torch.tensor(Nx)/2),torch.ceil(torch.tensor(Nx)/2),1)*(1/(Nx*deltaX)),dim=0)
        # y = np.expand_dims(np.arange(np.ceil(-Ny/2),np.ceil(Ny/2),1)*(1/(Ny*deltaY)),axis=1)
        y = torch.unsqueeze(torch.arange(torch.ceil(-torch.tensor(Ny)/2),torch.ceil(torch.tensor(Ny)/2),1)*(1/(Ny*deltaY)),dim=1)
        
        # print(x.shape)
        # print(y.shape)
        # y_new = np.repeat(y,Nx,axis=1)
        y_new = y.repeat(1, Nx)
        # x_new = np.repeat(x,Ny,axis=0)
        x_new = x.repeat(Ny,1)
        # print(y_new.shape)
        # print(x_new.shape)
        
        kp = torch.sqrt(y_new**2+x_new**2)
        term=k**2-kp**2
        term=np.maximum(term,0) 
        phase = torch.exp(1j*2*torch.pi*z*np.sqrt(term))
        # return torch.from_numpy(np.concatenate([np.real(phase)[np.newaxis,:,:,np.newaxis], np.imag(phase)[np.newaxis,:,:,np.newaxis]], axis = 3))
        return torch.cat([torch.real(phase).reshape(1,phase.shape[0],phase.shape[1],1), torch.imag(phase).reshape(1,phase.shape[0],phase.shape[1],1)], dim = 3)

    def roll_n(self, X, axis, n):
        f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    def batch_fftshift2d(self, x):
        real, imag = torch.unbind(x, -1)
        for dim in range(1, len(real.size())):
            n_shift = real.size(dim)//2
            if real.size(dim) % 2 != 0:
                n_shift += 1  # for odd-sized images
            real = self.roll_n(real, axis=dim, n=n_shift)
            imag = self.roll_n(imag, axis=dim, n=n_shift)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

    def batch_ifftshift2d(self,x):
        real, imag = torch.unbind(x, -1)
        for dim in range(len(real.size()) - 1, 0, -1):
            real = self.roll_n(real, axis=dim, n=real.size(dim)//2)
            imag = self.roll_n(imag, axis=dim, n=imag.size(dim)//2)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)
    
    def complex_mult(self, x, y):
        real_part = x[:,:,:,0]*y[:,:,:,0]-x[:,:,:,1]*y[:,:,:,1]
        real_part = real_part.unsqueeze(3)
        imag_part = x[:,:,:,0]*y[:,:,:,1]+x[:,:,:,1]*y[:,:,:,0]
        imag_part = imag_part.unsqueeze(3)
        return torch.cat((real_part, imag_part), 3)
    
    def TV(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,1:,:,:])
        count_w = self._tensor_size(x[:,:,1:,:])
        h_tv = torch.pow((x[:,1:,:,:]-x[:,:h_x-1,:,:]),2).sum() #gradient in horizontal axis
        w_tv = torch.pow((x[:,:,1:,:]-x[:,:,:w_x-1,:]),2).sum() #gradient in vertical axis
        return 0.01*2*(h_tv/count_h+w_tv/count_w)/batch_size
    
    def forward(self,x,y,z= torch.tensor(5000.) ):  #x: output holo y:captured holo z:distance
        x = x.squeeze(2)
        y = y.squeeze(2)
        x = x.permute([0,2,3,1])
        y = y.permute([0,2,3,1])
        
        self.z = z.squeeze().cpu()
        self.z = self.z.cpu()
        self.prop = self.propagator(self.Nx,self.Ny,self.z,self.wavelength,self.deltaX,self.deltaY)
        self.prop = self.prop.cuda()
        
        temp_x=torch.view_as_complex(x.contiguous())
          
       
        
        # cEs = self.batch_fftshift2d(torch.fft(x,3,normalized=True))
        
        cEs = self.batch_fftshift2d(torch.view_as_real (torch.fft.fftn(temp_x, dim=(0,1,2), norm="ortho")))
        
        cEsp = self.complex_mult(cEs,self.prop)
        
        # S = torch.ifft(self.batch_ifftshift2d(cEsp),3,normalized=True)
        
        temp = torch.view_as_complex(self.batch_ifftshift2d(cEsp).contiguous())
        S = torch.view_as_real(torch.fft.ifftn(temp, dim=(0,1,2), norm="ortho") )
        
        
        Se = S[:,:,:,0]
        
        loss = torch.mean(torch.abs(Se-torch.sqrt(y[:,:,:,0])))/2#torch.mean(torch.abs(Se-y[:,:,:,0]))/2#
        return loss


    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



```

