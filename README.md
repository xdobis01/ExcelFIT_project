# ExcelFIT Submission ID 35 Automated Human Recognition

modif

This code allows you to iterate over folder of images to detect location of human faces and their subsequent facial analysis. Facial analysis comprises of age, gender and emotion prediction  

## Getting Started

Please download folder cor_weights and ret_weights to your project directory, from https://vutbr-my.sharepoint.com/:f:/g/personal/xdobis01_vutbr_cz/EjVoLb2eScVClf0Akp89u7kB_OrZ8VWC6zk7aIk7ECQScg?e=gUAMUh

Update 12.4.

New video data set can be downloaded in folders cor_LSTM_train,cor_LSTM_test containing 68 and 35 unique sequences in image form. 

New examples of inference on test video data set images can be donwloaded in cor_RESULTS_comparison directory.

Update 3.5.

Greatly expanded training data set for LSTM training.

Video for ExcelFIT, showcasing solution and imroved age prediction with LSTM layers.

### Prerequisites

Using pip install following prerequisites, pip will download everything else needed automatically. If you want to do it manually, see full list in file prerequisities.txt

```
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python
pip install keras
pip install tensorflow
pip install natsort
```

### Directory inference 

Directory with images and directory to save results is mandatory input. Optional inputs are --cpu CPU (default) or --cuda GPU use, CORAL weights training dataset (AFAD,CACD,default MORPH,UTK), --RetinaFace_arch either default MobileNet or ResNet

```
python Recognition_net.py --data_folder C:\Python\TEST\testData\Excelobr --save_folder ./TEST/ --RetinaFace_arch MobileNet --CORAL_weights UTK --device cpu

```
### Examples 

Folder Examples, for all original authors CORAL weights data sets. Performance of weights is not consistent, retraining CORAL weights and combining it with temporal information in video is next goal of this work.

New examples are in folder cor_RESULTS_comparison, they need to be downloaded first, or download and predict test video data set folder cor_LSTM_test with each script (paths to images in cor_LSTM_test and to save directory cor_RESULTS_comparison are in scripts by default).

### Update 12.4.

New scripts,weights,video data set and examples were added.
* **New cor_weights** cor_weights have now newUTK (1-100) weights and their LSTM modified version
* **New cor_src** cor_src has now two new architectures one for each new CORAL weight
* **Recognition_net_newUTK.py** uses weights trained on whole age range (1-100) of UTK data set, therefore has no option to choose other CORAL weights (original authors implementation could not cover age 100, so new architecture==>new script was needed)
* **Recognition_net_LSTM.py** uses weights of Recognition_net_newUTK script, but architecture was modified with two LSTM layers which were then independently trained on video data set. Only used on video in form of image time series. First 19 images do not have age prediction. Prediction can have bad results passages between 2 persons. Usable only on one person per image.
* **New video data set** training data set in folder cor_LSTM_train and test data set in cor_LSTM_test, created from several videos of SoulPancake channel on Youtube https://www.youtube.com/playlist?list=PLzvRx_johoA-07L3F3FeTw5TrM7IY_kWb
* **New examples** in folder cor_RESULTS_comparison, there are 3 folders containing cor_LSTM_test images analyzed by scripts Recognition_net.py-Authors_weights, Recognition_net_newUTK.py-Original, Recognition_net_LSTM.py-Modified

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

