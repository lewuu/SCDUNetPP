# SCDUNet++ for Landslide Mapping Using Multi-Channel Remote Sensing Data  

📄 This repository contains the official implementation of our paper: **Landslide Mapping Based on a Hybrid CNN-Transformer Network and Deep Transfer Learning Using Remote Sensing Images with Topographic and Spectral Features Implementation for SCDUNet++.**  

📖 View paper [here](https://doi.org/10.1016/j.jag.2023.103612).  

👉 Here, we provide the pytorch implementation of SCDUNet++.  

## Requirements  

🖥️ This repo has been tested on Windows 10, Python 3.8, PyTorch 1.13.0, and CUDA 11.6. To setup the required modules, please run:  

```bash
pip install -r requirements.txt
```

## Installation  

💾 Clone this repository to your local machine.  

```bash
git clone https://github.com/lewuu/SCDUNetPP
cd SCDUNetPP
```

## Dataset preparation  

1️⃣ Create the directory 📁 `data/luding/DATA/img` and put the ***images*** in it. 🖼️  
2️⃣ Create the directory 📁 `data/luding/DATA/label` and put the ***labels*** in it. 🏷️  
3️⃣ Create the directory 📁 `data/luding/DATA/config` and put `train.txt`, `val.txt` and `test.txt` in it. 📝 The text files contain indexes of images and labels as follows (if the images are stored in TIF format):  

```
img1.tif,label1.tif
img2.tif,label2.tif
...,...
```

🔧 In the file `configs/config_case_luding.py`, you have the flexibility to adjust the parameters based on your dataset requirements. For example, you can modify the *num_classes*, *input_shape*, and *in_channels* to suit your needs.  

## Train  

✨ To initiate training, simply run the `train.py`. The weights for each epoch will be saved in the `checkpoints/luding/scdunetpp/loss_yyyy_mm_dd_hh_mm_ss` directory.  

## Test  

✨ Select a weight for testing from the model weights saved during each epoch of training. Put it in `checkpoints/Potsdam/scdunetpp` directory, and change *ckpt_test* in `configs/config_case_luding.py` to the name of the selected weights file.  

✨ Finally, run `test.py` to start testing.  

## Citation  

If you find this repo useful for your research, please consider citing the paper as follows:  

```
@article{WU2024103612,
title = {Landslide mapping based on a hybrid CNN-transformer network and deep transfer learning using remote sensing images with topographic and spectral features},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {126},
pages = {103612},
year = {2024},
issn = {1569-8432},
doi = {https://doi.org/10.1016/j.jag.2023.103612},
url = {https://www.sciencedirect.com/science/article/pii/S1569843223004363},
author = {Lei Wu and Rui Liu and Nengpan Ju and Ao Zhang and Jingsong Gou and Guolei He and Yuzhu Lei}
}
```
