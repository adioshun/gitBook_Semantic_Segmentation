<div style="text-align: center"><iframe width="560" height="315" src="https://youtu.be/_GfPYLNQank" frameborder="0" allowfullscreen></iframe> </div>

# Localization and Detection 
> [동영상](https://youtu.be/_GfPYLNQank), [강의자료](http://cs231n.stanford.edu/slides/2016/winter1516_lecture8.pdf)

기존 VGG, GoogLeNet, ResNet등이 좋은 성과를 보이고 있으나 이들은 주로 Classification용이다. 

새로운 연구 분야 중에 Localization and Detection 이 있다. (일부 Classification Network도 가능)

>  Localization and Detection 의 용어 적인 차이는 Localization 는 이미지에서 하나의 물체의 위치를 탐지 하는것, Detection 는 여러 물체의 위치를 탐지 하는것 

||Classification|Localization|
|-|-|-|
|Input|Image|Image|
|Output|Class label|Box in the image (x, y, w, h)|
|Evaluation metric|Accuracy|Intersection over Union|

###### [참고] 활용가능한 데이터셋

![](http://i.imgur.com/lVcAT2z.png)



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

> Detection as Regression이 가능한가? : 이미지에 여러(variabl) 객체가 있으면 Output도 다양해(1개의 위치 정보, n개의 위치정보) 지므로 어려움 
> - 해결책?? : Detection as Classification (0 , 1로 Output이 고정됨)

### 2.1 Detection as Classification #1
- Problem: Need to test many positions and scales
- Solution: If your classifier is fast enough, just do it

#### A. Histogram of Oriented Gradients

![](http://i.imgur.com/qG5ZuCW.png)
Dalal and Triggs, “Histograms of Oriented Gradients for Human Detection”, CVPR 2005

#### B. Deformable Parts Model (DPM)

![](http://i.imgur.com/BmiN0sP.png)

Felzenszwalb et al, “Object Detection with Discriminatively 


### 2.2 Detection as Classification #2

- Problem: Need to test many positions and scales,
+ and use a computationally demanding classifier (CNN)
- Solution: Only look at a tiny subset of possible positions (CNN은 Cost가 높기 때문에 일부영역만 선택적으로 탐색)

###### Region Proposal 

- Find “blobby” image regions that are likely to contain objects
- “Class-agnostic” object detector
- Look for “blob-like” regions

![](http://i.imgur.com/5gwoExY.png)

> 정확도가 높지 않고, 분류 기능이 없는 Object Detector를 이용하여서 물체가 있을듯한 위치 미리 찾아서 이후 작업 수행 

#### A. Region Proposal : Selective Search 

- 픽셀들간의 유사성을 중심으로 비슷한 색깔, 질감을 중심으로 영역 선정 

- Bottom-up segmentation, merging regions at multiple scales

![](http://i.imgur.com/5xw9kcl.png)

###### [참고] 기타 Detection Proposals 

![](http://i.imgur.com/IggJTP6.png)
 Hosang et al, “What makes for effective detection proposals?”, PAMI 2015
 
#### B. Region Proposal : R-CNN

- CNN과 Region Proposal을 합친 아이디어 . 

![](http://i.imgur.com/ZnruDOn.png)

단점 
- Slow at test-time: need to run full forward pass of CNN for each region proposal
- SVMs and regressors are post-hoc: CNN features
not updated in response to SVMs and regressors
- Complex multistage training pipeline

###### Step 1. Train (or download) a classification model for ImageNet (AlexNet)

![](http://i.imgur.com/DHG7VX6.png)

###### Step 2. Fine-tune model for detection
- Instead of 1000 ImageNet classes, want 20 object classes + background
- Throw away final fully-connected layer, reinitialize from scratch
- Keep training model using positive / negative regions from detection images

![](http://i.imgur.com/PZPgswk.png)

###### Step 3. Extract features
- Extract region proposals for all images
- For each region: warp to CNN input size, run forward through CNN, save pool5
features to disk
- Have a big hard drive: features are ~200GB for PASCAL dataset!

![](http://i.imgur.com/thhaBMk.png)

###### Step 4. Train one binary SVM per class to classify region features

![](http://i.imgur.com/8ttvHoY.png)

###### Step 5. bbox regression 
For each class, train a linear regression model to map from cached features to offsets to GT boxes to make up for “slightly wrong” proposals

![](http://i.imgur.com/EcvTNYv.png)

#### C. Region Proposal : Fast R-CNN

R-CNN의 속도 단점 해결 : Extract Region과 CNN의 위치 바꿈 (cf. 슬라이딩 위도우의 아이디어 유사)
Region of Interest pooling 이 중심 아이디어 

> [CS231n Lecture 8의 Fast R-CNN부분](https://youtu.be/_GfPYLNQank?t=42m4s0)

||![](http://i.imgur.com/1xYICdA.png)|![](http://i.imgur.com/eJdUJfn.png)|
|-|-|-|
|Problem|#1 Slow at test-time due to independent forward passes of the CNN|#2 Post-hoc training: CNN not updated in response to final classifiers and regressors <br><br> #3: Complex training pipeline|
|Solution|Share computation of convolutional layers between proposals for an image|Just train the whole system end-to-end all at once!|

###### Step 1.

![](http://i.imgur.com/t60O2iN.png)

###### Step 2. 

![](http://i.imgur.com/q9VMaWu.png)

###### Step 3.

![](http://i.imgur.com/pAl3V7Q.png)

###### Step 4.

![](http://i.imgur.com/5QXYx02.png)

###### Step 5.

![](http://i.imgur.com/wkYNq5v.png)



#### D. Region Proposal : Faster R-CNN
- Fast R-CNN문제 : Test-time speeds don’t include region proposals
- Faster R-CNN 해결책 : Just make the CNN do region proposals too!

##### 가. 기존과 다른점 

![](http://i.imgur.com/ewg7Oxw.png)
- Insert a Region Proposal Network (RPN) after the last convolutional layer

- RPN trained to produce region proposals directly; no need for external region proposals!

- After RPN, use RoI Pooling and an upstream classifier and bbox regressor just like Fast R-CNN

##### 나. Region Proposal Network (RPN)

![](http://i.imgur.com/Jw5cuHl.png)

Slide a small window on the feature map

Build a small network for:
- classifying object or not-object, and
- regressing bbox locations

Position of the sliding window provides localization information with reference to the image

Box regression provides finer localization information with reference to this sliding window

![](http://i.imgur.com/t7k2zSC.png)

Use N anchor boxes at each location

Anchors are **translation invariant**: use the same ones at every location

Regression gives offsets from anchor boxes 

Classification gives the probability that each (regressed) anchor shows an object

##### 다. Training

![](http://i.imgur.com/cgpFgpj.png)

In the paper: Ugly pipeline
- Use alternating optimization to train RPN, then Fast R-CNN with RPN proposals, etc.
- More complex than it has to be

Since publication: Joint training! One network, four losses
- RPN classification (anchor good / bad)
- RPN regression (anchor -> proposal)
- Fast R-CNN classification (over classes)
- Fast R-CNN regression (proposal -> box)

![](http://i.imgur.com/ugTk83C.png)

>2016년 현재 최고의 성능은 ResNet 101 + Faster R-CNN + some extras 합친것 
> - He et. al, “Deep Residual Learning for Image Recognition”, arXiv 2015

#### E. Region Proposal : YOLO

> 정확도는 R-CNN보다 느리나, 속도가 빠름(Real-time)

![](http://i.imgur.com/4OK8i7N.png)

Divide image into S x S grid

Within each grid cell predict:
- B Boxes: 4 coordinates + confidence
- Class scores: C numbers

Regression from image to 7 x 7 x (5 * B + C) tensor

Direct prediction using a CNN




