# VidFuncta_public

This repo contains the official Pytorch implementation of the paper *VidFuncta*.
 

## Data
-The Echonet Dynamic dataset can be downloaded [here](https://aimi.stanford.edu/datasets/echonet-dynamic-cardiac-ultrasound)

-The Breast Ultrasound Video dataset can be downloaded [under this link](https://github.com/jhl-Det/CVA-Net/tree/main](https://github.com/jhl-Det/CVA-Net/tree/main )  

-The BEDLUS dataset of lung ultrasound videos can be required [here]( https://www.dropbox.com/scl/fi/ztzo9pt8i81ad1uz73x27/BEDLUS-data-instructions.pdf?rlkey=jxndke0vesoyg8wn28wydlwdz&e=1&dl=0 )

A mini-example how the data needs to be stored can be found in the folder *data*. 


### Training of the Meta-Model
- To run the training of the SIREN MLP, run...
- 
The trained models will be stored in a folder *logs*.

### Check the Performance of the Pretrained Binarizing Autoencoder
- For the BRATS2020 dataset, run
`python3 reconstruct.py  ....`



### Inference and saving of the modulation vectors

- To run the training of the Bernoulli diffusion model on the BRATS2020 dataset, run
`python ./Bernoulli_Diffusion/scripts/latent_train.py --sampler bld  --dataset brats --data_dir './data/brats/training'  --codebook_size 128 --nf 32  --img_size 256 --batch_size 36 --latent_shape 1 32 32 --ch_mult 1 2 2 4 --n_channels=4 --ae_load_dir ./logs/binaryae_brats --ae_load_step 00000`
- To run the training of the Bernoulli diffusion model on the OCT2017 dataset, run
 `python  ./Bernoulli_Diffusion/scripts/latent_train.py --sampler bld  --dataset OCT --data_dir './data/OCT/training'  --codebook_size 128 --nf 32  --img_size 256 --batch_size 36 --latent_shape 1 32 32 --ch_mult 1 2 2 4 --n_channels=1 --ae_load_dir ./logs/binaryae_OCT --ae_load_step 00000`
 
 The trained Bernoulli diffusion models will be stored in a folder *results*.


## Detailed Results

## Comparing Methods
### PocovidNet
### Res2+1D

### MedFuncta


### Spatial Functa


