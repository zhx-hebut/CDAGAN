# Cross-Domain Attention-Guided Generative Data Augmentation for Medical Image Analysis with Limited Data

This repository is the official implementation of the CIBM paper [Cross-Domain Attention-Guided Generative Data Augmentation for Medical Image Analysis with Limited Data](https://www.sciencedirect.com/science/article/pii/S001048252301209X). 

## Usage
### Requirements

To install the necessary requirements:

```setup
pip install -r requirements.txt
```
### Datasets
To download the necessary datasets, visit the following links:
- [BraTS 2020](https://www.med.upenn.edu/cbica/brats-2020/)
- [TCIA](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

### Cross-Domain Data Generation

To train the CDAGAN model, execute those commands:

```train
cd CDAGAN
bash train_cda.sh
```

To generate data for data augmentation on medical image analysis tasks based on the trained CDAGAN model, execute those commands:

```infer
cd CDAGAN
bash infer_cda.sh
```

### Classification

To perform data augmentation on classification tasks, execute those commands:
```
cd ResNet
bash classification.sh 
```

To evaluate the performance of the trained classification model, execute those commands:
```
cd ResNet
python infer.py
```

### Segmentation

To perform data augmentation on segmentation tasks, execute those commands:
```
cd UNet
python train.py
```

To evaluate the performance of the trained segmentation model, execute those commands:
```
cd UNet
python test.py
```

## Citation

If you find CDAGAN helpful for your research, please consider citing:

    @article{xu2023cross,
    title={Cross-domain attention-guided generative data augmentation for medical image analysis with limited data},
    author={Xu, Zhenghua and Tang, Jiaqi and Qi, Chang and Yao, Dan and Liu, Caihua and Zhan, Yuefu and Lukasiewicz, Thomas},
    journal={Computers in Biology and Medicine},
    pages={107744},
    year={2023},
    publisher={Elsevier}
    }

For any issues or queries, please contact chang.qi@student.tuwien.ac.at.