# This directory is for model testing only.
#### 1. First, make sure the dataset which the model is trained on is the sanme as the dataset name in common.py
#### 2. Run ```make test MODEL='where the model is saved'(dataset name is included in the experiment name) BN=True/False(whether to use adaBN) SAVE=0(number of prediction images to save)```
For example, the commmand could be ```make test model='/usr/xtmp/satellite/train_models/xh.deeplab.mobilenet.vegas/epoch463.pth' BN=True SAVE=0```
#### 3. The result will be saved in 'train_log' as a json file.
