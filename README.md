# LampMark: Proactive Deepfake Detection via Training-Free Landmark Perceptual Watermarks

Source code implementation of our [paper](https://dl.acm.org/doi/10.1145/3664647.3680869) accepted to Proceedings of the 32nd ACM International Conference on Multimedia (MM 2024).

## Datasets used in this project

LampMark is trained using CelebA-HQ and tested on CelebA-HQ and LFW. We do not own the datasets, and they can be downloaded from the official webpages.
* [Download CelebA-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* [Download LFW](https://vis-www.cs.umass.edu/lfw/)

After splitting the image data following the official document of CelebA-HQ, the folder should be named as ```dataset_celeba_hq/``` and placed under ```image_data/```. For the cross-dataset evaluation under a balanced ratio, LFW is processed such that one for each identity is adopted. The directory should look like the following:
```
LampMark
└── image_data/
    ├── dataset_celeba_hq/
    │   ├── train/
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   └── ...
    │   ├── val/
    │   │   ├── 1000.jpg
    │   │   └── ...         
    │   └── test/
    │       ├── 10008.jpg
    │       └── ...    
    └── lfw/
        └── test/
            ├── AJ_Cook_0001.jpg
            └──...
```

In this project, the landmarks with 106 points are extracted via [Face++](https://console.faceplusplus.com/documents/13207488) with paid services. We directly provide the watermarks that we used for training, validation, and testing, which can be found in ```watermark_data\```. The watermarks are stored as ```.npy``` files such that the file names match with the image file names. 


## Train the model from scratch

The model is trained following the configuration files located in ```configuration/```.

We pre-train the framework against the common manipulation **Jpeg(50)** and then fine-tune it against **SimSwap**. The model is pre-trained following settings in ```configuration/pretrain.json```, and fine-tuned following settings in ```configuration/tune_deepfake.json```.

We use ```main.py``` to pre-train and fine-tune the model by calling ```train_common()``` and ```tune_deepfake()```. Switch the mode by commenting out the unneeded one for pre-training and fine-tuning. Simply modify the configuration file and run 
```
python main.py
```


## Test the model

The model is tested following the configuration files located in ```configuration/```.

We use ```configuration/test_common.json``` to test the watermarks against all benign manipulations and derive the watermark recovery accuracies. We use ```configuration/test_deepfake.json``` to test the watermarks against Deepfake manipulations and derive the watermark recovery accuracies. 

We use ```main.py``` to test the model against the desired adversaries (e.g., benign manipulations, SimSwap, InfoSwap). 


## Use Deepfake models for LampMark to defend against

LampMark is trained against SimSwap, and tested against seven Deepfake models including SimSwap. Since we don't own the source code, we recommend downloading and placing the model source code and weights by yourself. The models should be placed under ```model/``` folder so that the classes in ```model/deepfake_manipulations.py``` can utilize the generative models. 

The source code can be found at the following links:
* [SimSwap (ACM MM 2020)](https://github.com/neuralchen/SimSwap)
* [InfoSwap (CVPR 2021)](https://github.com/GGGHSL/InfoSwap-master)
* [UniFace (ECCV 2022)](https://github.com/xc-csc101/UniFace)
* [E4S (CVPR 2023)](https://github.com/e4s2022/e4s/tree/main)
* [StarGAN (CVPR 2020)](https://github.com/clovaai/stargan-v2)
* [StyleMask (FG 2023)](https://github.com/StelaBou/StyleMask)
* [HyperReenact (ICCV 2023)](https://github.com/StelaBou/HyperReenact)

