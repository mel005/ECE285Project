# CycleGan
This is a Pytorch Implementation for learning an image-to-image translation. The original paper is called [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf). Instead of using default generator network, we use two other network namely DnCNN and UNet. Detail structure can be found in this [paper](https://www.google.com). Some of our result is shown below:
![alt text](https://github.com/menglaili/ECE285Project/blob/master/CycleGan/img/result_Unet-2.png)



Requirements
============
Custom python module: image_tool.py, nntools.py



Code organization
=================
model.py         ---The net which is just the model downloaded from Internet

DnCNNmodel.py    ---The net with DnCNN as generator

Unetmodel.py     ---The net with Unet as generator

dataset.py       ---The .py file help to load the image

image_pool.py    ---The .py file which buffer the images(downloaded from Internet)

train_CycleGAN.ipynb ---The .ipynb file which run the experiments to obtain model

Demo_CycleGAN.ipynb    ---Use the trained model to do testing, 'the trained model is stored in output2

MyCycleGan-DnCNN.ipynb ---Training record with DnCNN as generator

MyCycleGan-Unet.ipynb  ---Training record with Unet as generator

