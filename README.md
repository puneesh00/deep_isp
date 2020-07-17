# Deep_ISP

## Pre-requisites
The code was written with Python 3.6.8 with the following dependencies:
* cuda release 9.0, V9.0.176
* tensorflow 1.12.0
* keras 2.2.4
* numpy 1.16.4
* scipy 1.2.1
* imageio 2.5.0
* skimage 0.15.0
* matplotlib 3.1.0
* cuDNN 7.4.1

This code has been tested in Ubuntu 16.04.6 LTS with 4 NVIDIA GeForce GTX 1080 Ti GPUs (each with 11 GB RAM).

## How to Use 
### Clone the repository:
```
git clone https://github.com/puneesh00/deep_isp.git
```
## Testing 

### Download weights

&nbsp; Download weights for the model, and place them in the cloned git repository. They can be found [here]().

#### To infer full resolution images, run the following command:

```
python infer_full.py -path (give full path to the repository) -w weights2_0191.h5 -dataset (path to full resolution raw images)
```
&nbsp; This will generate the output images in a folder `results` (default name) in the git repository.

#### To infer cropped frames, run the following command:

```
python infer.py -path (give full path to the repository) -w weights2_0191.h5 -dataset (path to cropped raw images) -res results_cropped 
```
&nbsp; This will generate the output images in a folder `results_cropped` in the git repository.


