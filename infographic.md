# The Modern History of Object Recognition — Infographic

> 출처 : [medium](https://medium.com/@nikasa1889/the-modern-history-of-object-recognition-infographic-aea18517c318), [pdf](https://drive.google.com/file/d/0B9SNmZvpaGj1NnNsbWhTZUxYSlU/view?usp=drivesdk)

## 1. Object Recognition Research Area 

![](http://i.imgur.com/w4D29jQ.png)

## 2. History 

![](http://i.imgur.com/PXoQ353.png)

## 3. Important CNN Concepts

### 3.1 Feature 

![](http://i.imgur.com/xEUlmtH.png)
(pattern, activation of a neuron, feature detector)

- A hidden neuron that is activated when a particular pattern (feature) is presented in its input region (receptive field).

- The pattern that a neuron is detecting can be visualized by 
    - optimizing its input region to maximize the neuron’s activation (deep dream),
    - visualizing the gradient or guided gradient of the neuron activation on its input pixels (back propagation and guided back propagation), 
    - visualizing a set of image regions in the training dataset that activate the neuron the most
    
### 3.2 Receptive Field

![](http://i.imgur.com/btEb1LA.png)
(input region of a feature)

The region of the input image that affects the activation of a feature. In other words, it is the
region that the feature is looking at.

Generally, a feature in a higher layer has a bigger receptive field, which allows it to learn to capture
a more complex/abstract pattern. 

The ConvNet architecture determines how the receptive field change layer by layer.

![](http://i.imgur.com/btEb1LA.png)