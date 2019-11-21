# This directory is for test model only.
#### 1. First, make sure the dataset name which the model is trained with is the sanme as the dataset name in common.py
#### 2. run ```make test MODEL='where the model is saved' BN=True/False(whether to use adaBN) SAVE=0(number of prediction images to save)```  
For example, the commmand could be ```make test model='/usr/xtmp/satellite/train_models/xh.deeplab.mobilenet.vegas/epoch463.pth' BN=True SAVE=0```
