# Transfer Learning

만약 이미지셋이 100만개보다 적다면 pre-train 모델을 사용하라!!!



![](http://sebastianruder.com/content/images/2017/03/andrew_ng_drivers_ml_success-1.png)

> "Transfer learning will be the next driver of ML success", Andrew NG, NIPS 2016


## 역사??
- 1. 1995년 NIPS workshop에서 "Learning to learn"이라는 주제로 소개
- 2. Domain Adaptation : knowledge transfer
- 3. Transfer learning

![](http://nmhkahn.github.io/assets/CNN-Practice/vggnet.png)

위 이미지와 같은 VGGNet을 어떤 도메인에 사용해야 할 경우가 있다고 생각해보자. 맨땅에 VGGNet을 학습시키기 위해서는 매우매우 많은 데이터가 필요하고, 학습하는데 걸리는 시간도 매우 긴 것은 사실이다 (2~3주 걸린 것으로 알고 있다). 하지만 VGGNet의 pre-train된 모델을 구할 수 있다면 (Caffe model zoo) 이를 갖다가 사용하면 문제가 매우 쉬워진다.

만약 적용해야 하는 도메인의 데이터셋이 작다면 VGGNet의 끝에 있는 FC-1000과 softmax 레이어를 지워버리고 내게 맞는 도메인 예를 들어 CIFAR-10이라면 FC-10, softmax로 바꾼다. 또 softmax 대신 SVM을 사용하거나 다른 loss function을 사용하거나.. 뭐 자기 마음이다. 그리고 네트워크를 학습 할 때 기존의 레이어는 pre-train된 모델의 weight, bias를 그대로 사용하고 추가한 레이어만 학습을 시킨다. 모든 레이어를 학습시키면 좋은 성능이 나올지는 모르지만 적은 데이터셋에서는 그렇게 효과적인지 의문이고 가장 큰 문제는 학습이 오래걸린다..

도메인의 데이터셋이 적당히 있다면 추가한 레이어와 그 위 몇개 레이어를 (그 위 maxpool과 conv 2개 정도?) fine-tune한다. 데이터가 많으면 많을 수록 fine-tune하는 레이어를 늘려도 괜찮고 데이터가 꽤 많으면 모든 레이어를 학습시키는 것도 고려할 수 있다.
fine-tune을 할 때 한가지 팁은 새로 추가한 레이어의 learning_rate는 기존 네트워크에서 사용한 것 보다 10배 작게, 중간 레이어들은 100배 작게 learning_rate를 사용하는 게 좋다고 한다.

뉴럴넷의 얕은 레이어(입력 이미지와 가까운 레이어)는 edge나 texture를 검출하는 등의 역할을 하는 이미지에 대해 매우 포괄적으로 사용 가능한 레이어이다. 반면에 깊은 레이어는 학습에 사용된 데이터셋에 specfic하기 때문에 얕은 레이어도 fine-tune하면 물론 좋지만, 꼭 그럴 필요는 없다.

## Transfer Learning Scenarios 분류

### 1. ConvNet as fixed feature extractor
Take a ConvNet pretrained on ImageNet, remove the last fully-connected layer (this layer’s outputs are the 1000 class scores for a different task like ImageNet), then treat the rest of the ConvNet as a fixed feature extractor for the new dataset. In an AlexNet, this would compute a 4096-D vector for every image that contains the activations of the hidden layer immediately before the classifier. We call these features CNN codes. It is important for performance that these codes are ReLUd (i.e. thresholded at zero) if they were also thresholded during the training of the ConvNet on ImageNet (as is usually the case). Once you extract the 4096-D codes for all images, train a linear classifier (e.g. Linear SVM or Softmax classifier) for the new dataset.

### 2. Fine-tuning the ConvNet
The second strategy is to not only replace and retrain the classifier on top of the ConvNet on the new dataset, but to also fine-tune the weights of the pretrained network by continuing the backpropagation. It is possible to fine-tune all the layers of the ConvNet, or it’s possible to keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network. This is motivated by the observation that the earlier features of a ConvNet contain more generic features (e.g. edge detectors or color blob detectors) that should be useful to many tasks, but later layers of the ConvNet becomes progressively more specific to the details of the classes contained in the original dataset. In case of ImageNet for example, which contains many dog breeds, a significant portion of the representational power of the ConvNet may be devoted to features that are specific to differentiating between dog breeds.

### 3. Pretrained models
Since modern ConvNets take 2-3 weeks to train across multiple GPUs on ImageNet, it is common to see people release their final ConvNet checkpoints for the benefit of others who can use the networks for fine-tuning. For example, the Caffe library has a Model Zoo where people share their network weights.

## Transfer Learning Scenarios 결정 기준 
> 출처 : http://nmhkahn.github.io/CNN-Practice

> 출처 : http://cs231n.github.io/transfer-learning, [[번역]](http://ishuca.tistory.com/entry/CS231n-Transfer-Learning-and-Finetuning-Convolutional-Neural-Networks-%ED%95%9C%EA%B5%AD%EC%96%B4-%EB%B2%88%EC%97%AD)

> CIFAR10의 60,000개의 데이터는 많다고 판단, CIFAR10 & ImageNet는 같은 도메인이라고 판단

### 1. 도메인이 기존 데이터셋과 비슷하고, 데이터가 적다
끝 레이어(top layer)에 도메인에 맞는 레이어를 추가하고 추가한 레이어만 학습한다.

- 데이터양이 적기 때문에 Fine Tune을 할경우 Overfitting될 우려가 있다. 
- 데이터가 기본 데이터셋과 비슷하므로 ConvNet의 Higher-Lever의 특징이 비슷하다고 가정 할수 있음 
- [결론] train a linear classifier on the CNN codes.


```

for layer in model.layers:
   layer.trainable = False
#Now we will be training only the classifiers (FC layers)

```
> [코드 출처](https://medium.com/towards-data-science/transfer-learning-using-keras-d804b2e04ef8)

### 2. 도메인이 기존 데이터셋과 비슷하고, 데이터가 많다
추가한 레이어와 몇개 레이어를 fine-tune 한다.

- 추가 데이터가 생긴것과 비슷
- 더 많은 자료를 가졌기 때문에, 전체 망을 통해 Fine-tune을 시도한다면 과적합 없는 더 신뢰를 가질 수 있다.
- More High layer의 일부를 적절 제거 한후 새로 합습 한다. (FC 무조건 재 학습)

```python
for layer in model.layers:
   layer.trainable = True
#The default is already set to True. I have mentioned it here to make things clear.
for layer in model.layers[:5]:
   layer.trainable = False.
# Here I am freezing the first 5 layers 
```

> [코드 출처](https://medium.com/towards-data-science/transfer-learning-using-keras-d804b2e04ef8)



    
### 3. 도메인이 기존 데이터셋과 매우 다르고, 데이터가 적다
큰일이다…..

- 데이터가 적기 때문에 only train a linear classifier하는것이 최선이다. 
- 데이터가 기존 데이터셋과 다르므로 
    - 추천 : train the SVM classifier from activations somewhere earlier in the network
    - 비추 : train the classifier form the top of the network(=more dataset-specific features)


### 4. 도메인이 기존 데이터셋과 매우 다르고, 데이터가 많다
많은 레이어를 fine-tune 한다.

- 충분한 데이터가 있으므로 그냥 처음부터 학습 하는게 좋을수 있다. 
- 그러나 현실에서는 전체 층에 대해서 Fine-Tune을 수행 한다. 

---
1. [도메인이 기존 데이터셋과 매우 다르]XS≠XTXS≠XT. The feature spaces of the source and target domain are different, e.g. the documents are written in two different languages. In the context of natural language processing, this is generally referred to as cross-lingual adaptation.

2. P(XS)≠P(XT)P(XS)≠P(XT). The marginal probability distributions of source and target domain are different, e.g. the documents discuss different topics. This scenario is generally known as domain adaptation.

3. YS≠YTYS≠YT. The label spaces between the two tasks are different, e.g. documents need to be assigned different labels in the target task. In practice, this scenario usually occurs with scenario 4, as it is extremely rare for two different tasks to have different label spaces, but exactly the same conditional probability distributions.

4. P(YS|XS)≠P(YT|XT)P(YS|XS)≠P(YT|XT). The conditional probability distributions of the source and target tasks are different, e.g. source and target documents are unbalanced with regard to their classes. This scenario is quite common in practice and approaches such as over-sampling, under-sampling, or SMOTE are widely used.

> 출처 : http://sebastianruder.com/transfer-learning/index.html

---

# 적용 분야 

1.  시뮬레이션을 통한 학습 : Objects in the simulation and the source look different
    - eg) 무인자동차 운전 (Self Driving Car)

2. Adapting to new domains: 

3. Transferring knowledge across language

# Transfer Learning Methods
- Using pre-trained CNN features
- Learning domain-invariant representations
- Making representations more similar
- Confusing domains


> 출처 : http://sebastianruder.com/transfer-learning/index.html

# Trasfer Learing 제약 
- 함부로 기존 네트워크를 변경 할수 없다. 하지만 일부 항목들은 가능한다(eg. 이미지 크기)
- Learging rate를 기본의 값보다 작게 가져라 [[출처]](http://ishuca.tistory.com/entry/CS231n-Transfer-Learning-and-Finetuning-Convolutional-Neural-Networks-%ED%95%9C%EA%B5%AD%EC%96%B4-%EB%B2%88%EC%97%AD)


#### Bottlenecking
- FC만 변경하여 학습 해도 시간이 많이 걸린다면 [Bottlenecking](https://medium.com/@galen.ballew/transferlearning-b65772083b47)기법 적용

- FC레이어가 없는 기존넷(VGG16)을 이용하여 `예측`한 값을 사용 
    - `bottleneck_features = model.predict_generator(generator,)
    - 학습보다 예측이 Back Propagation이 없으므로 속도가 빠름 


####  PathNet
인공지능은 전이학습을 통해 이전의 학습에서 배운 지식을 완전히 새로운 과제에 활용할 수 있음.

- 이전의 지식을 활용함으로써 인공지능은 새로운 과제에 새로이 신경망을 구축할 필요가 없으며, 전이학습은 과제 이행의 질과 시간적 효율성을 높여줌.

전이학습은 신경망을 구축하고 학습하는데 있어서 가장 큰 과제이며, DeepMind는 PathNet을 통해 이를 해결하기 위해 노력중임.
- PathNet이란 SGD(stochastic gradient descent)와 유전 선발방법(genetic selection method)을 사용하여 학습한 신경망 네트워크를 뜻함.
- PathNet은 여러 층(layer)의 모듈로 이루어져 있으며, 각각의 모듈은 다양한 형태의 신경망임(예: 나선형(convolutional), 피드포워드(feedforward) 또는 반복되는(recurrent) 형태).

학습을 하기에 앞서, 모듈에 대한 정의를 해야 함(L = 층수(number of layers), N = 하나의 층이 가질 수 있는 최대 모듈수).
- 마지막 층은 복잡하게 이루어져 있으며, 다른 과제와 함께 공유되지 않음.

- A3C(Advantage Actor-critic)를 활용하는 마지막 층은 가치함수(value function)와 정책평가(policy evaluation)을 나타냄.

- 모듈에 대한 정의가 이루어진 후, P 유전자형(또는 경로)가 네트워크에 생성됨. 각각의 유전자형 평가를 통해 최적화된 경로를 파악하고, 이 경로를 통해 학습을 힘.

- 경로의 학습은 백 프로퍼게이션(back propagation)을 사용한 SGD(stochastic gradient descent)를 통해 이루어지며, 한 번에 하나의 경로를 통해 실행됨.

과제를 학습한 후 네트워크는 최적 경로의 모든 매개변수(parameters)를 수정하며, 모든 매개변수는 다시 초기화 됨.

- A3C는 이전지식이 지워지지 않도록 이전과제의 최적경로가 수정되지 않도록 함.

> 출처 : [미래금융연구센터](http://www.kcft.or.kr/2017/02/2120)

---
[A Survey on Transfer Learning - SJ Pan et al. 2010](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)