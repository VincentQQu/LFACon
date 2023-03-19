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
<pre>
@article{qu2023lfacon,
  title={LFACon: Introducing Anglewise Attention to No-Reference Quality Assessment in Light Field Space},
  author={Qu, Qiang and Chen, Xiaoming and Chung, Yuk Ying and Cai, Weidong},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2023},
  publisher={IEEE}
}

@article{qu2021light,
  title={Light field image quality assessment with auxiliary learning based on depthwise and anglewise separable convolutions},
  author={Qu, Qiang and Chen, Xiaoming and Chung, Vera and Chen, Zhibo},
  journal={IEEE Transactions on Broadcasting},
  volume={67},
  number={4},
  pages={837--850},
  year={2021},
  publisher={IEEE}
}
</pre>

### To Be Updated
