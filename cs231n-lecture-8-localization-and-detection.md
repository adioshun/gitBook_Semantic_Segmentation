<div style="text-align: center"><iframe width="560" height="315" src="https://youtu.be/_GfPYLNQank" frameborder="0" allowfullscreen></iframe> </div>


# Localization and Detection 

 기존 VGG, GoogLeNet, ResNet등이 좋은 성과를 보이고 있으나 이들은 주로 Classification용이다. 

새로운 연구 분야 중에 Localization and Detection 이 있다. (일부 Classification Network도 가능)

>  Localization and Detection 의 용어 적인 차이는 Localization 는 이미지에서 하나의 물체의 위치를 탐지 하는것, Detection 는 여러 물체의 위치를 탐지 하는것 

||Classification|Localization|
|-|-|-|
|Input|Image|Image|
|Output|Class label|Box in the image (x, y, w, h)|
|Evaluation metric|Accuracy|Intersection over Union|

 > 활용가능한 테스트 데이터 : ImageNet

## 1 .  Localization

### 1.1   Localization as Regression

간단하면서도 성능이 좋다. 

![](http://i.imgur.com/YByjh5S.png)

###### Step 1: Train (or download) a classification model (AlexNet, VGG, GoogLeNet)

###### Step 2: Attach new fully-connected “regression head” to the network
- Conv layer 이후 : Overfeat, VGG
- Last FC layer 이후 : DeepPose, R-CNN

###### Step 3: Train the regression head only with SGD and L2 loss

###### Step 4: At test time use both heads

### 1.2 Sliding Window (Overfeat)

- Run classification + regression network at multiple locations on a high resolution image

- Convert fully-connected layers into convolutional layers for efficient computation

- Combine classifier and regressor predictions across all scales for final prediction


![](http://i.imgur.com/mbc5mL3.png)

OverfeatL Alexnet기반, Winner of ILSVRC 2013 localization challenge

특징 : Efficient sliding window by converting fully connected layers into convolutions (Fully connected Layer를 Conv. Layer로 바꾸어서 작업 )

### 1.3 Localization 네트워크 성능 

![](http://i.imgur.com/ER1L4cv.png)


## 2 . (Object) Detection 