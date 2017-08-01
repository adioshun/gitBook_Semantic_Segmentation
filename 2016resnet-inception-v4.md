> “Identity Mappings in Deep Residual Networks", 2016

Inception-V1(2014)
Inception-V2 = Inception-V1 +  BN(batch normalization) 
Inception-V3(2015) = nception-V2 + onvolution factorization + label smoothing + Auxiliary classifier
Inception-V4(2016) = 
Inception-ResNet

> 추후 다시 살펴 보기 : 라온피플 블로그[Class49](http://laonple.blog.me/220752877630), [Class50](http://laonple.blog.me/220752877630) 



# Inception-V4

## 1. 개요 





기존 ResNet에 pre-activation 개념 적용 

## 2. 특징 

### 2.1 Residual Network Module

![](http://i.imgur.com/f75jugS.png)

$$ 
x_{l+1} = f(h(x_l) + F(x_l))
$$

- f(x) : activation(ReLU) 함수
- h(x) : identity 함수
- F(x) : Residual 함수



