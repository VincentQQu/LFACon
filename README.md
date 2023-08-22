# LFACon
This is the repository for paper *"LFACon: Introducing Anglewise Attention to No-Reference Quality Assessment in Light Field Space"* published on [IEEE TVCG](https://ieeexplore.ieee.org/document/10049721) and paper *"Light Field Image Quality Assessment With Auxiliary Learning Based on Depthwise and Anglewise Separable Convolutions"* published on [IEEE TBC](https://ieeexplore.ieee.org/abstract/document/9505016).



============= model weights to be updated ======================

### New: Predict Quality Scores Using LFACon
Due to GitHub's size limitations, the complete project, including model weights, has been uploaded to Google Drive.

To obtain quality scores using LFACon, follow these steps:

1. Download and unzip the entire folder (~1 GB, mean and std to normalise LFIs contribute a lot) from [Google Drive](https://drive.google.com/drive/folders/1Bh-sxVQCevkhkRCx0eMCeeFhWXHKAQqv?usp=sharing).
2. Place the Light Field Image (LFI) you want to assess into one of the dataset folders: `./Dataset/Win5-LID`, `./Dataset/SMART`, or `./Dataset/MPI-LFA`. Two sample LFIs have been provided in each folder for your convenience.
3. Modify line 1 of `constants.py` to reflect the dataset folder you placed the LFI in (i.e., `Win5-LID`, `SMART`, or `MPI-LFA`).
4. Run `python3 app.py`.
5. The results will be displayed in the terminal and saved to `./Datasets/quality_predictions/`.

Please note that Win5-LID, SMART, and MPI-LFA use different scoring systems:

- Win5-LID employs a Mean Opinion Score (MOS) ranging from 1 to 5.
- SMART uses the Bradley-Terry (BT) scoring system, which typically ranges from -13 to 0.
- MPI-LFA utilizes the Just Objectionable Difference (JOD) score, with a normal range of -9 to 0.

In all the scoring systems mentioned above, higher values indicate better quality.

Keep in mind that **LFACon will estimate the quality based on different scoring systems** depending on the dataset folder you place your LFI in.


### Requirements
matplotlib==3.3.0,
numpy==1.23.5,
pandas==1.0.5,
Pillow==9.5.0,
scipy==1.10.1,
seaborn==0.10.1,
tensorflow==2.10.1



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
  title={LFACon: Introducing Anglewise Attention to No-Reference Quality Assessment in Light Field Space}, 
  year={2023},
  volume={29},
  number={5},
  pages={2239-2248},
  doi={10.1109/TVCG.2023.3247069}
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
