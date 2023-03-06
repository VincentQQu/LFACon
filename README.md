# LFACon
This is the repository for paper "LFACon: Introducing Anglewise Attention to No-Reference Quality Assessment in Light Field Space" published in [IEEE TVCG](https://ieeexplore.ieee.org/document/10049721).

To be updated



### Requirements
matplotlib==3.3.0 \n
numpy==1.18.5
pandas==1.0.5
Pillow==9.4.0
scipy==1.5.0
seaborn==0.10.1
tensorflow==2.2.1



### File Descriptions
xpreprocess.py: essentials to convert massive raw LFI dataset into tidy trainable train-test-splitted data (including some data augmentation functions)
xmodels.py: definiation of LFACon and its layers
constants.py: storing global variables
xtrains.py: training models with checkpoints
utils.py: utilities for training such as batch generator and evaluators.

