# openpifpaf_apollocar3d

## Setup
Download openpifpaf, switch to dev branch and install it with:

`pip3 install --editable '.[dev,train,test]`

`pip3 install pandas`

`pip3 install itermplot`

(in case CUDA 9 as driver: 
` pip install torch==1.7.0+cu92 torchvision==0.8.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html`)

* Download openpifpaf_animalpose, and create the following directories:
    * `mkdir data`
    * soft link output directory, which needs to be called outputs
    * soft link to animalpose dataset
    * create apollo-coco directory with `images/train`, `images/val`, `annotations` subdirectories and soft link them.
    
    
## Preprocess Dataset
`python -m openpifpaf_animalpose.voc_to_coco` 

## Show poses
`python -m openpifpaf_apollocar3d.utils.constants`

## Pretrained models
TODO

## Train
TODO

## Everything else
All pifpaf options and commands still hold, please check the 
[DEV guide](https://vita-epfl.github.io/openpifpaf/dev/intro.html)
