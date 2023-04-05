# cloth_shape_estimation


## Dependencies
1) DiffCloth

#### Clone the repo:
`git clone --recursive https://github.com/friolero/DiffCloth.git`

#### Build the Python binding:
```
cd DiffCloth
python setup.py install --user
```

2) Pytorch3D

The complete official installation can be refered [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). This is the installation without using [Conda](https://docs.conda.io/en/latest/).

#### Install cuda 11.6
```
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
sudo chmod +x cuda_11.6.0_510.39.01_linux.run
sudo ./cuda_11.6.0_510.39.01_linux.run
```

#### Install Nvidia cub
```
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
```

### Install dependencies and Pytorch3d 
```
pip install -U fvcore iopath 
pip install scikit-image matplotlib imageio plotly opencv-python
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu116_pyt1130/download.html
```

## Steps

#### Generate perturbed cloth data
Mode 0 for randomized perturbation trajectory; and mode 1 uses Bezier curve to generate the perturbation path from a randomly identified control vertex to another vertex position within a specified distance range.
`python3 launch_cloth_randomization.py 0/1`

#### Cloth shape estimiation via differentiable rendering
`python3 cloth_shape_estimation.py `
