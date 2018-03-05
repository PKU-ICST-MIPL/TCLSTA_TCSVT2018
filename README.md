## Introduction
This is the source code of our TCSVT 2018 paper "Two-stream Collaborative Learning with Spatial-Temporal Attention for Video Classification", Please cite the following paper if you use our code.

Yuxin Peng, Yunzhen Zhao, and Junchao Zhang, "Two-stream Collaborative Learning with Spatial-Temporal Attention for Video Classification", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), DOI: 10.1109/TCSVT.2018.2808685, 2018.[【pdf】](http://59.108.48.34/tiki/download_paper.php?fileId=20187)

## Dependency
Our code is based on [Caffe](https://github.com/BVLC/caffe), all the dependencies are the same as Caffe. 

The provided caffe code ```caffe-rc3-lstm/``` is modified on the [rc3](https://github.com/BVLC/caffe/tree/rc3) version.

Note that for the implementation of the LSTM layer in ```caffe-rc3-lstm/```, please refer to [Junhyuk Oh's implementation](https://github.com/junhyukoh/caffe-lstm).

The proposed TCLSTA also uses the Pre-trained ResNet-50 model with batch normalization, which can be downloaded at [Caffe model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo#imagenet-pre-trained-models-with-batch-normalization), download this model and put it in example/semihash/Pre_trained folder.

## Data Preparation
Here we use [UCF101](http://crcv.ucf.edu/data/UCF101.php) dataset for an example, download the UCF101 dataset, and put the extracted frames and optical flow images in ```dataset/UCF101/UCF101_jpegs_256/``` and ```dataset/UCF101/UCF101_tvl1_flow/``` folders, respectively.

It's recommended to use the [Christoph Feichtenhofer's toolkit](https://github.com/feichtenhofer/gpu_flow) to compute optical flow.

## Usage

1. The training of spatial-temporal attention model.
For the stable convergence of spatial-temporal attention model, we take the following training steps:
  1) Train ```Connection network``` and ```Spatial-level attention network``` jointly and get the spatial attention model.
  2) Train ```Temporal-level attention network``` based on the obtained spatial attention model, with freezing the weights of ```Connection network``` and ```Spatial-level attention network```.
  3) Train the spatial-temporal attention model jointly based on the obtained model by step 2).
2. The training of static-motion collaborative model.
3. Testing.
