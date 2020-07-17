# Deep_ISP

## How to Use 
### Clone the repository:
```
git clone https://github.com/estija/LIME.git
```
## Testing 

### Download weights

&nbsp; Download weights for the model, and place them in the cloned git repository. They can be found [here]().

#### To infer full resolution images, run the following command:

```
python infer_full.py -path (give full path to the repository) -w weights2_0191.h5 -dataset (path to full resolution raw images)
```
&nbsp; This will generate the output images in a folder results (default name) in the git repository.

#### To infer cropped frames, run the following command:

```
python infer.py -path (give full path for to the repository) -w weights2_0191.h5 -dataset (path to cropped raw images) -res results_cropped 
```
&nbsp; This will generate the output images in a folderresults_cropped in the git repository.


