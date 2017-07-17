# Deep Learning Is Not Good Enough, We Need Bayesian Deep Learning for Safe AI

> 출처 : [Alex Kendall](http://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/)

> [SegNet and Bayesian SegNet Tutorial](https://github.com/alexgkendall/SegNet-Tutorial)

## 1. 개요 
Understanding what a model does not know is a critical part of many machine learning systems. Unfortunately, today’s deep learning algorithms are usually unable to understand their uncertainty. These models are often taken blindly and assumed to be accurate, which is not always the case. For example, in two recent situations this has had disastrous consequences.
- In May 2016 we tragically experienced the first fatality from an assisted driving system. According to the manufacturer’s blog, “Neither Autopilot nor the driver noticed the white side of the tractor trailer against a brightly lit sky, so the brake was not applied.”
- In July 2015, an image classification system erroneously identified two African American humans as gorillas, raising concerns of racial discrimination. See the news report here.

> Model이 알지 못하는 것을 예측하는 것은 머신러닝의 큰 문제점 중에 하나 이다. 불행이도, 일반적인 딥러닝 알고리즘들은 자신의 불확실성(Uncertainty)를 알지 못한다. 이러한 문제점으로 최근 아래 두가지 문제가 발생 하였었다. 
> - 2016: 자율 주행 차량이 트랙터의 옆면의 밝은 부분을 도로로 인식하여 발생한 교통 사고 
> - 2015: 구글 이미지 검색에서 고릴라를 검색하면 흑인의 사진이 나온 사건 

And I’m sure there are many more interesting cases too! If both these algorithms were able to assign a high level of uncertainty to their erroneous predictions, then each system may have been able to make better decisions and likely avoid disaster.

> 만일 위 두 알고리즘이 `able to assign a high level of uncertainty to their erroneous predictions` 있었다면 이러한 문제점을 해결 할수 있었을 것이다. 

It is clear to me that understanding uncertainty is important. So why doesn’t everyone do it? The main issue is that traditional machine learning approaches to understanding uncertainty, such as Gaussian processes, do not scale to high dimensional inputs like images and videos. To effectively understand this data, we need deep learning. But deep learning struggles to model uncertainty.

> uncertainty가 이렇게 중요 한데 왜 연구가 활성화 되지 않았을까? 이유는 기존의 머신러닝 알고리즘들이 uncertainty를 이해 하는 방식(eg. Gaussian processes)은 고차원(이미지, 비디오)데이터 처리에 적합하지 않기 때문이다. 

In this post I’m going to introduce a resurging field known as Bayesian deep learning (BDL), which provides a deep learning framework which can also model uncertainty. BDL can achieve state-of-the-art results, while also understanding uncertainty. I’m going to explain the different types of uncertainty and show how to model them. Finally, I’ll discuss a recent result which shows how to use uncertainty to weight losses for multi-task deep learning. 
> 본 투고글에서 uncertainty를 모델링 할수 알수 있는 `Bayesian deep learning (BDL)`에 대하여 알아 보겠다.
> - Different types of uncertainty
> - Uncertainty를 이용하여 weight losses할수 있는 방법

The material for this blog post is mostly taken from my two recent papers: 
- What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? Alex Kendall and Yarin Gal, 2017. (.pdf)
- Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. Alex Kendall, Yarin Gal and Roberto Cipolla, 2017. (.pdf)

> 본 투고글의 내용은 위 두 논문의 내용을 기반으로 하였다. 


#### 예시 : Depth estimation에 uncertainty가 중요한 이유 

![](http://i.imgur.com/pyUh2Jn.png)
An example of why it is really important to understand uncertainty for depth estimation. 
- The first image is an example input into a Bayesian neural network which estimates depth, as shown by the second image. 
- The third image shows the estimated uncertainty. 
You can see the model predicts the wrong depth on difficult surfaces, such as the red car’s reflective and transparent windows. 
Thankfully, the Bayesian deep learning model is also aware it is wrong and exhibits increased uncertainty.

> 두번째 이미지를 BNL을 이용하여 깊이 측정을 한것이다. 3번째 그림은 estimated uncertainty를 나타낸다. 빨간차의 반사 부분이나 투명한 창문은 잘 구분하지 못한다. 
> 하지만, BNL을 사용하면 이러한 문제점을 해결 할수 있다. 



## 2. Types of uncertainty
The first question I’d like to address is what is uncertainty? There are actually different types of uncertainty and we need to understand which types are required for different applications. I’m going to discuss the two most important types – epistemic and aleatoric uncertainty.
> uncertainty란 무었일까? 사실 uncertainty에는 두가지 종류가 있다. 

### 2.1 Epistemic uncertainty(지식의 불확실성_
Epistemic uncertainty captures our ignorance about which model generated our collected data. This uncertainty can be explained away given enough data, and is often referred to as model uncertainty. 
> `Epistemic uncertainty`는 정보 부족에 의해서 발생한다. __model uncertainty__라고도 불리우는데 충분한 데이터가 있으면 uncertainty을 알수 있다. 

Epistemic uncertainty is really important to model for:
- Safety-critical applications, because epistemic uncertainty is required to understand examples which are different from training data,
- Small datasets where the training data is sparse.

> Epistemic uncertainty은 아래 두가지 상황을 모델링 할때 중요하다. 
> - 자율 주행 등의 Safety-critical서비스 
> - 학습 데이터가 Sparse된 적은양의 데이터 

> 추가 : 지식의 불확실성을 처리하기 위한 기존의 접근법으로는 베이지안 접근법(Bayesian approach)이 널리 사용되고 있다.





### 2.2 Aleatoric uncertainty(우연적 불확실성)
Aleatoric uncertainty captures our uncertainty with respect to information which our data cannot explain. 
- For example, aleatoric uncertainty in images can be attributed to occlusions (because cameras can’t see through objects) or lack of visual features or over-exposed regions of an image, etc. 
It can be explained away with the ability to observe all explanatory variables with increasing precision. 

> `Aleatoric uncertainty`는 데이터로도 설명할수 없는 정보에 대한 불확실성에 기인한다. (=고유의 변동성으로 인해 존재)
> - 예를 들어 카메라로는 측적이 어려운  occlusions나 lack of visual features 나 over-exposed regions가 있다.

> all explanatory variables에 대한 자세한 관찰을 통해 해결 할수 있다. 


For example, aleatoric uncertainty in images can be attributed to occlusions (because cameras can’t see through objects) or lack of visual features or over-exposed regions of an image, etc. It can be explained away with the ability to observe all explanatory variables with increasing precision. 
Aleatoric uncertainty is very important to model for:
- Large data situations, where epistemic uncertainty is mostly explained away,
- Real-time applications, because we can form aleatoric models as a deterministic function of the input data, without expensive Monte Carlo sampling.

> Aleatoric uncertainty은 아래 두가지 상황을 모델링 할때 중요하다. 
> - Large data
> - Real-time applications

> 추가 : 우연적 불확실성을 처리하기 위한 전통적 접근법으로는 과거자료에 근거한 확률론적 분석, 즉 빈도주의적 접근법(frequentist approach)이 주로 사용 


We can actually divide aleatoric into two further sub-categories:
> `Aleatoric uncertainty`는 다시 두개의 서브 카테고리로 나눌수 있다. 

#### 2.2.1 Data-dependant or Heteroscedastic uncertainty
Data-dependant or Heteroscedastic uncertainty is aleatoric uncertainty which depends on the input data and is predicted as a model output.

>입력데이터에 의존적이고, 모델 아웃풋으로 예측 가능??(is predicted as a model output)

#### 2.2.1 Task-dependant or Homoscedastic uncertainty 
Task-dependant or Homoscedastic uncertainty is aleatoric uncertainty which is not dependant on the input data. It is not a model output, rather it is a quantity which stays constant for all input data and varies between different tasks. 

> 입력데이터에 무의존적이고, 

It can therefore be described as task-dependant uncertainty. 

Later in the post I’m going to show how this is really useful for multi-task learning.


## 3. Bayesian deep learning
> 이번장에서는 Bayesian deep learning을 이용해서 uncertainty을 측정할수 있는 모델을 만들어 보겠다. 

Bayesian deep learning is a field at the intersection between deep learning and Bayesian probability theory. It offers principled uncertainty estimates from deep learning architectures. These deep architectures can model complex tasks by leveraging the hierarchical representation power of deep learning, while also being able to infer complex multi-modal posterior distributions.

> BDL은 딥러닝과 베이지안 확률 이론의 중간 정도에 위치하고 있다. 주요 기능은 deep learning architectures의 uncertainty를 측정 하는 것이다.  
  
Bayesian deep learning models typically form uncertainty estimates by either placing distributions over model weights, or by learning a direct mapping to probabilistic outputs. 

> BDL이 Uncertainty 측정은 placing distributions over model weights하거나 learning a direct mapping to probabilistic outputs을 통해서 가능하다. 


In this section I’m going to briefly discuss how we can model both epistemic and aleatoric uncertainty using Bayesian deep learning models.

> 본 장에서는 BDL모델을 이용하여 epistemic and aleatoric uncertainty를 모데링 하는 법을 알아 보곘다.

Firstly, we can model Heteroscedastic aleatoric uncertainty just by changing our loss functions. 
> 간단히 loss functions을 수정하여 이분산성 우연적 불확실성을 모델링 한다. 

Because this uncertainty is a function of the input data, we can learn to predict it using a deterministic mapping from inputs to model outputs. 
> 왜냐 하면 이 불확실성은 입력 데이터에 대한 함수 이기 때문에 deterministic mapping from inputs to model outputs을 통해 알수 있다. 


For regression tasks, we typically train with something like a Euclidean/L2  loss: $$ Loss =\parallel y- \hat{y} \parallel _2 $$ . To learn a Heteroscedastic uncertainty model, we simply can replace the loss function with the following:

||기존|BDL|
|-|-|-|
|Regression|$$Loss =\parallel y- \hat{y} \parallel _2 $$|$$ Loss = \frac{\parallel y- \hat{y} \parallel _2}{2\sigma^2} + \log\sigma^2$$|

- where the model predicts a mean $$ \hat{y} $$ and variance $$\sigma^2 $$. 
- if the model predicts something very wrong, then it will be encouraged to attenuate the residual term, by increasing uncertainty $$\sigma^2 $$. 
- However, the $$\log\sigma^2$$ prevents the uncertainty term growing infinitely large. 
This can be thought of as learned loss attenuation.

Homoscedastic aleatoric uncertainty can be modelled in a similar way, however the uncertainty parameter will no longer be a model output, but a free parameter we optimise.
> Homoscedastic aleatoric uncertainty도 비슷한 방법으로 모델링 할수 있다. 하지만, 불확실성 파라미터는 no longer be a model output, but a free parameter we optimise.

On the other hand, epistemic uncertainty is much harder to model. This requires us to model distributions over models and their parameters which is much harder to achieve at scale. 
> 반면에, epistemic uncertainty는 더 모델링 하기 어렵다. epistemic uncertainty는 distributions over models해야 하고, parameters is harder to achieve at scale하다. 

A popular technique to model this is Monte Carlo dropout sampling which places a Bernoulli distribution over the network’s weights.

> 자주 사용되는 방법은 `Monte Carlo dropout sampling`기술 이다. 이것은 `Bernoulli distribution`을 네트워크 weight에 위치 시킨다. 

In practice, this means we can train a model with dropout. Then, at test time, rather than performing model averaging, we can stochastically sample from the network with different random dropout masks. 

The statistics of this distribution of outputs will reflect the model’s epistemic uncertainty.

In the previous section, I explained the properties that define aleatoric and epistemic uncertainty. One of the exciting results in our paper was that we could show that this formulation gives results which satisfy these properties. 

Here’s a quick summary of some results of a monocular depth regression model on two datasets:

These results show that when we train on less data, or test on data which is significantly different from the training set, then our epistemic uncertainty increases drastically. 

However, our aleatoric uncertainty remains relatively constant, which it should because it is tested on the same problem with the same sensor.

