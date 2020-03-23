# ExcelFIT_project Automated Human Recognition

This code allows you to iterate over folder of images to detect location of human faces and their subsequent facial analysis. Facial analysis comprises of age, gender and emotion prediction  

## Getting Started

Please download folder cor_weights and ret_weights to your project directory, from https://vutbr-my.sharepoint.com/:f:/g/personal/xdobis01_vutbr_cz/EjVoLb2eScVClf0Akp89u7kBEUPUdeRi1XYo64tL7lELWg?e=JEaHHX

### Prerequisites

Using pip install following prerequisites, pip will download everything else needed automatically. If you want to do it manually, see full list in file prerequisities.txt

```
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python
pip install keras
pip install tensorflow
```

### Directory inference 

Directory with images and directory to save results is mandatory input. Optional inputs are --cpu CPU (default) or --cuda GPU use, CORAL weights training dataset (AFAD,CACD,default MORPH,UTK), --RetinaFace_arch either default MobileNet or ResNet

```
python Recognition_net.py --data_folder C:\Python\TEST\testData\Excelobr --save_folder ./TEST/ --RetinaFace_arch MobileNet --CORAL_weights UTK --device cpu

```
### Examples 

Are in folder Examples, for all CORAL weights data sets. Performance of weights is not consistent, retraining CORAL weights and combining it with temporal information in video is next goal of this work.

## Authors

* **xdobis01 - Lukáš Dobiš** 

* **Authors of RetinaFace model** initalization code and weigths https://github.com/biubug6/Pytorch_Retinaface
* **Authors of CORAL model** initalization code and weigths https://github.com/Raschka-research-group/coral-cnn
* **Authors of Gender and Emotion models** initalization code and weigths https://github.com/kbsriharsha/FacialEmotionAnalysis

    
## License

This project is licensed under the Creative Commons license - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* doc. Ing. Radim Kolář Ph.D. for advices on how to write article
* Authors of used architectures for making their work available
* FIT for allowing me to participate in ExcelFIT

