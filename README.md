# LFACon
This is the repository for paper *"LFACon: Introducing Anglewise Attention to No-Reference Quality Assessment in Light Field Space"* published on [IEEE TVCG](https://ieeexplore.ieee.org/document/10049721) and paper *"Light Field Image Quality Assessment With Auxiliary Learning Based on Depthwise and Anglewise Separable Convolutions"* published on [IEEE TBC](https://ieeexplore.ieee.org/abstract/document/9505016).

To be updated



### Requirements
matplotlib==3.3.0,
numpy==1.18.5,
pandas==1.0.5,
Pillow==9.4.0,
scipy==1.5.0,
seaborn==0.10.1,
tensorflow==2.2.1



### File Descriptions
* **xpreprocess.py**: essentials to convert massive raw LFI dataset into tidy trainable train-test-splitted data (including data augmentation methods)
* **xmodels.py**: definiation of LFACon and its layers
* **constants.py**: storing global variables
* **xtrains.py**: training models with checkpoints
* **utils.py**: utilities for training such as batch generator and evaluators.


### To Cite
@article{qu2023lfacon,<br />
  title={LFACon: Introducing Anglewise Attention to No-Reference Quality Assessment in Light Field Space},<br />
  author={Qu, Qiang and Chen, Xiaoming and Chung, Yuk Ying and Cai, Weidong},<br />
  journal={IEEE Transactions on Visualization and Computer Graphics},<br />
  year={2023},<br />
  publisher={IEEE}<br />
}

@article{qu2021light,
  title={Light field image quality assessment with auxiliary learning based on depthwise and anglewise separable convolutions},<br />
  author={Qu, Qiang and Chen, Xiaoming and Chung, Vera and Chen, Zhibo},<br />
  journal={IEEE Transactions on Broadcasting},<br />
  volume={67},<br />
  number={4},<br />
  pages={837--850},<br />
  year={2021},<br />
  publisher={IEEE}<br />
}


### To Be Updated
