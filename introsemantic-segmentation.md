![](https://i.imgur.com/h8SOkWp.png)





# Image Segmentation

- Pixel Level Segmentation 
    - DeepLap V3
- Instance level Segmentation (Object Detection쪽에 가까움)
    - Mask R-CNN
    - SegNet

## 1. CNNs

### 1.1 Fully Convolution Network
특징 
- End-to-end, Pixel-to-pixel prediction
- Backwards convolution for up-sampling
- Per-pixel multinomial logistic loss

단점 
- Fixed size receptive field
- Too simple structure to get detailed features

### 1.2 Deconvolution Network
특징 
- Combining unpooling, deconvolution(with crop), and Relu
- Reconstruction of the detailed structure of an object in finer resolution
- Batch-normalization

단점
- Difficult to learn
- Still lose spatial information

### 1.3 [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
특징 
- Do not use unpooling(only up-convolution)
- Skip-connection(with concat)
- Do not have fully connected layer
- Elastic deformation

단점 
- Didn’t use batch-norm
- VGG is not the best solution for feature extracting

> Medical Data용?

### 1.4 [Deep contextual networks](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/11789)

특징
- Auxiliary connection, classifier
- Ensemble
- Lower memory consumption

단점 
- Didn’t use batch-norm
- VGG is not the best solution for feature extracting

### 1.5 [FusionNet](https://arxiv.org/pdf/1612.05360.pdf)

특징
- Skip-connection(with summation)
- Residual block(shortcut connection)
- Elastic deformation

단점 : 메모리 

### 1.6 [Pyramid Scene Parsing Net](https://arxiv.org/pdf/1612.01105.pdf)

특징 
- Pre-trained FCN with ResNet(1/8 sized feature map)
- Pyramid pooling & 1x1 cone
- Bilinear interpolation
- Avg pooling is better than Max pooling


## 2. RNNs

### 2.1 [Multi-Dimensional RNNs](https://arxiv.org/pdf/0705.2011.pdf)
특징 
- GOD GRAVES!!
- 1D RNNs(Bi-directional RNNs) couldn’t explain images well
- Need to access to the surrounding context in all directions
- N-dimensional data : At least 2^(N) hidden layers
- The input layer is size 3(RGB) or 1(Gray) or patch and the output layer(softmax) is size of classes



## 3. GANs


https://www.slideshare.net/HyungjooCho2/image-segmentation-hjcho



