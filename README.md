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


