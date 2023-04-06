# cloth_shape_estimation


## Dependencies

### 1) DiffCloth

> #### Clone the repo:
> `git clone --recursive https://github.com/friolero/DiffCloth.git`
> 
> #### Build the Python binding:
> ```
> cd DiffCloth
> python setup.py install --user
> ```

### 2) Pytorch3D

> The complete official installation can be refered [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). This is the installation without using [Conda](https://docs.conda.io/en/latest/).
> 
> #### Install cuda 11.6
> ```
> wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
> sudo chmod +x cuda_11.6.0_510.39.01_linux.run
> sudo ./cuda_11.6.0_510.39.01_linux.run
> ```
> 
> #### Install Nvidia cub
> ```
> curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
> tar xzf 1.10.0.tar.gz
> export CUB_HOME=$PWD/cub-1.10.0
> ```
> 
> ### Install dependencies and Pytorch3d 
> ```
> pip install -U fvcore iopath 
> pip install scikit-image matplotlib imageio plotly opencv-python
> pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
> pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu116_pyt1130/download.html
> ```

## Usage

#### 1. Generate perturbed cloth data as .obj files
`python3 cloth_randomization.py -mode 1 -output_dir cloth_project -n_output 5000 -seed SEED -n_openmp_thread N_PARALLEL_THREAD`

> Wavefront Data will be saved under $DIFFCLOTH_PATH/output/$args.output_dir.
> * Mode 0 for randomized perturbation trajectory; 
> * Mode 1 uses Bezier curve to generate the perturbation path from a randomly identified control vertex to another vertex position within a specified distance range.


#### 2. Data driven deformation estimation with deform-net 
Create rendered image from perturbed cloth obj files and partition data into training, evaluation and testing set. This may takes 1-2 hours.
`python3 data_utils.py`

Start deform_net training with:
`python3 deform_net_train.py`

Perform inference with trained deform_net :
`python3 deform_net_predict.py -m model.ckpt -obj PATH_TO_OBJ`


#### 3. Cloth deformation optimization via differentiable rendering
`python3 cloth_deform_estimation.py`

> * Step (2) is optional to run step (3)
