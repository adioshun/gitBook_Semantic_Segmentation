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

- The region of the input image that affects the activation of a feature. 

- In other words, it is the region that the feature is looking at.

- Generally, a feature in a higher layer has a bigger receptive field, which allows it to learn to capture a more complex/abstract pattern. 

- The ConvNet architecture determines how the receptive field change layer by layer.

### 3.3 Feature Map

![](http://i.imgur.com/wRi3zbP.png)

(a channel of a hidden layer)

- A set of features that created by applying the same feature detector at different locations of an input map in a sliding window fashion (i.e. convolution). 

- Features in the same feature map have the same receptive size and look for the same pattern but at different locations. 

- This creates the spatial invariance properties of a ConvNet.

### 3.4 Feature Volume

![](http://i.imgur.com/8p72KhI.png)
(a hidden layer in a ConvNet)

- A set of feature maps, each map searches for a particular feature at a fixed set of locations on the input map.

- All features have the same receptive field size.

### 3.5 Fully connected layer as Feature Volume

![](http://i.imgur.com/oiVYeDH.png)

Fully connected layers (fc layers - usually attached to the end of a ConvNet for classification) with k hidden nodes can be seen as a $$1 \times 1 \times k$$ feature volume. 

This feature volume has one feature in each feature map, and its receptive field covers the whole image. 

The weight matrix W in an fc layer can be converted to a CNN kernel.

Convolving a kernel $$w \times h \times k$$ to a CNN feature volume $$w \times h \times d$$ creates a $$1 \times 1 \times k$$ feature volume (=FC layer with k nodes). 

Convolving a $$1 \times 1 \times k$$ filter kernel to a $$1 \times 1 \times d$$ feature volume creates a $$1 \times 1 \times k$$ feature volume. 

Replacing fully connected layers by convolution layers allows us to apply a ConvNet to an image with arbitrary size.

