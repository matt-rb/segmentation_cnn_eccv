# Image Segmentation Deep-Binary
general image segmentation. Deep-Net to train segments regradless to the object categories.
</br>
In preparation for ECCV 2016.


#### Prerequisites

* Install [`Torch7`](http://torch.ch/docs/getting-started.html) 
* Have at least Cuda 7.0
* Install [`CuDNN`](https://github.com/soumith/cudnn.torch)
* Install [`matio-ffi`](https://github.com/soumith/matio-ffi.torch) 

<a name="alex.dirstructure"></a>
## Directories ##

* *model* : 
binary torch alexnet models. to download models run [getmodels.sh](#models).

* *scripts* : 
includes script and functions for lua, matlab and python.</br>
`lua` shared script folder. for import in new script and use functions :
```lua
require("scripts.lua.common")
require("scripts.lua.utils")
require("scripts.lua.create_models")
```
`common` includes common directories addresses, required libraries and also 
`utils` and `create_models` functions. </br>
`utils` includes frequently used functions such as load image, converters, etc.
</br>
`create_models` includes functions to create torch models.

`matlab` matlab script and function folder.

* *data* : 
test sample data and images (input data).

<a name="alex.models"></a>
## Models ##

to download pretrained models from dropbox run `getmodels.sh` script from 
folder `/segmentation/model` : </br>
```
$ bash ./getmodels.sh 
```
`getmodels.sh` parameters:

```
$ bash ./getmodels.sh -h
Downloading torch models
-m --model name of model (default all)
-d --dir download directory (default current)
----------------- Model List -------------------------
0 - all
1 - alex_std
2 - alex_fullconv_992
3 - alex_fullconv_1000
```

<a name="alex.scripts"></a>
## Scripts ##

* *th_model_manipulation* :
First remove the softmax layer, and then fc7 from fully_conv alexnet. creating two new models
 `th_model_full_conv_fc7` and `th_model_full_conv_fc6`, make them ready to extract fc6, fc7 output feats.

* *th_extract_feats* :
Extract `fc6` and `fc7` output feats for PASCAL dataset and save to `.mat` file.