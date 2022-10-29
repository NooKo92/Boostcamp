# 1. Introduction to Deep Learning
## 1) AI, ML, DL의 관계
$$\mathsf{Deep\ Learning} \subset \mathsf{Machine\ Learning} \subset \mathsf{Artificial\ Intelligence}$$
- AI: Mimic human intelligence
- ML: Data-driven approach
- DL: Neural Network

## 2) Key Components Deep Learning
### (1) Data
- Data는 모델이 학습할 수 있는 무언가를 뜻한다.
- Data는 다음과 같은 문제의 유형들에 따라 달라진다.
    - classification
    - semantic segmentation (No objects, just pixels)
    - detection
    - pose estimation
    - visual QnA

### (2) Model
- Model은 data를 변환하는 방법을 의미한다.
- 대표적으로 다음과 같은 모델들이 있다.
    - AlexNet
    - GoogLeNet
    - ResNet
    - DenseNet
    - LSTM
    - DeepAutoEncoders
    - GAN

### (3) Loss Function
- Loss Function은 모델의 성능을 정량화한다.
- Loss Function은 문제의 유형에 따라 달라진다.
- 대표적으로 다음과 같은 세가지의 loss function이 사용된다.
    - Regression Task: $MSE = \cfrac{1}{N} \sum_{i=1}^N \sum_{d=1}^D \left( y_i^{(d)}-\hat{y}_i^{(d)} \right)^2$
    - Classification Task: $CE = -\cfrac{1}{N} \sum_{i=1}^N \sum_{d=1}^D y_i^{(d)}\log\hat{y}_i^{(d)}$
    - Probabilistic Task: $MLE = \cfrac{1}{N} \sum_{i=1}^N \sum_{d=1}^D \log\mathcal N \left( y_i^{(d)};\hat{y}_i^{(d)}, 1 \right)$

### (4)Optimization Algorithm
- Optimization Algorithm은 loss function을 최소화하기 위해 parameter를 조정한다.

# 2. Neural Networks & Multi-Layer Perceptron
## 1) Neural Netwoks
- Neural Networks는 동물 뇌를 구성하는 생물학적 신경망에서 영감을 얻은 계산 시스템이다. 

### (1) Linear Neural Networks
- 가장 간단한 선형 모델 $\hat{y} = wx + b$를 예로 들면, parameter w와 b는 $\cfrac{\partial loss}{\partial w} = 0\ ,\cfrac{\partial loss}{\partial b} = 0$으로 구할 수 있고 행렬형태로는 $\mathbf{\hat{y}} = \mathbf{W}^T\mathbf{x} + \mathbf{b}$로 나타낼 수 있다.

### (2) Non-linear Neural Networks
- 만약 x와 y사이에 h라는 변수가 있다면 다음과 같이 표현할 수 있다. $\mathbf{\hat{y}} = \mathbf{W}_2^T\mathbf{h} = \mathbf{W}_2^T\mathbf{W}_1^T\mathbf{x}$
- 그러나 단순히 input과 output사이의 변수를 늘리는 것은 여전히 선형모델이기 때문에 의미가 없다.
- 이러한 이유로, 선형모델에 비선형성을 부여하기 위해 activation function이 필요하고 이를 추가한 식은 $\mathbf{\hat{y}} = \mathbf{W}_2^T\mathbf{h} = \mathbf{W}_2^T\rho\mathbf{W}_1^T\mathbf{x}$가 된다.

## 2) Multi-Layer Perceptron
- Neural Networks를 여러층 쌓은 것을 Multi-Layer Perceptron이라 한다.
- input x와 output y 사이에 여러개의 hidden layer(h)가 있는 구조이고 식으로는 $\mathbf{\hat{y}} = \mathbf{W}_3^T\mathbf{h}_2 = \mathbf{W}_3^T\rho\left(\mathbf{W}_2^T\mathbf{h}_1\right) = \mathbf{W}_3^T\rho\left(\mathbf{W}_2^T\rho\left(\mathbf{W}_1^T\mathbf{x}\right)\right)$ 와 같이 표현할 수 있다.

# 3. Optimization
## 1) Important Concepts in Optimization
### (1) Generalization
- 학습된 모델이 새로운 데이터에 대해 얼마나 좋은 성능을 보여주는지 나타내는 지표
- $\mathsf{Generalization\ gap} = \mathsf{Test\ error} - \mathsf{Training\ error}$
- Generalization gap이 작은 것이 Test error 또한 작은 것을 의미하지는 않는다.

### (2) Underfitting vs. Overfitting


### (3) Cross-Validation
- 독립적인 test data에 대해 모델이 일반적으로 잘 작동되도록 하기 위해 사용하는 방법
- 주로 overfitting을 막기 위해 사용한다.
- 가장 간단한 방법으로는 train data를 n개로 나눈 후 n-1개로 학습시키고 나머지 하나로 test 하는 것을 n번 반복하는 방법이 있다.

### (4) Bias and Variance
- Bias: 예측값과 실제 정답의 차이의 평균
- Variance: 예측값이 얼마나 다양하게 분포하는지를 나타내는 지표
- 수학적으로 Bias와 Variance를 동시에 줄일 수는 없다.

### (5) Bootstrapping
- 복원 추출을 사용하는 test 방법

### (6) Bagging vs. Boosting
- Bagging (Bootstrapping aggregating)
    - Boostrapping을 이용해 여러 모델을 병렬적으로 학습시키는 방식
- Boosting
    - 분류하기 어려운 특정 sample에 초점을 맞춘 방식
    - 각 모델들이 순차적으로 학습하면서 이전 모델의 약점을 보완해 나간다.

## 2) Practical Gradient Descent Methods
- Batch-size Matters: Batch size가 크면 sharp minimum에 도달하고 작으면 flat minimum에 도달한다.

### (1) (Stochastic) Gradient Descent
$$W_{t+1} = W_t - \eta g_t$$ 
- $\eta$: learning rate
- 적절한 $\eta$를 찾는 것이 어렵다.

### (2) Momentum
$$a_{t+1} = \beta a_t + g_t\ $$
- $a_{t+1}$: accumulation, $\beta$: momentum

$$W_{t+1} = W_t - \eta a_{t+1}$$
- 이전에 구한 g를 포함해서 계산

### (3) Nesterov Accelerated Gradient
$$a_{t+1} = \beta a_t + \nabla \mathcal{L}\left(W_t - \eta\beta a_t\right)$$
- $\nabla\mathcal{L}\left(W_t - \eta\beta a_t\right)$: Lookahead gradient

$$W_{t+1} = W_t - \eta a_{t+1}$$

### (4) Adagrad
- gradient가 크면 learning rate가 작아지고 gradient가 작으면 learning rate가 커지는 방식

$$W_{t+1} = W_t - \cfrac{\eta}{G_t + \epsilon}g_t$$
- $G_t$: sum of gradient squares, $\epsilon$: for numerical stability
- $t \rightarrow \infty$이면 $G_t$가 커지기 때문에 학습이 더뎌진다.

### (5) Adadelta
- Adagrad에서 gradient sum의 범위에 제한을 두어 학습속도를 조절하는 방식

$$G_t = \gamma G_{t-1} - (1-\gamma)g_t^2$$
- $G_t$: EMA(Exponential Moving Average) of gradient squares

$$W_{t+1} = W_t - \cfrac{\sqrt{H_{t-1} + \epsilon}}{\sqrt{G_t + \epsilon}}g_t$$

$$H_t = \gamma H_{t-1} - (1-\gamma)\left(\triangle W_t\right)^2$$
- learning rate이 없기 때문에 많이 사용되지 않는다.

### (6) RMSprop
- Geoff Hinton이 그의 강의에서 소개한 방법

$$G_t = \gamma G_{t-1} - (1-\gamma)g_t^2$$
$$W_{t+1} = W_t - \cfrac{\eta}{\sqrt{G_t + \epsilon}}g_t$$
- $\eta$: stepsize

### (7) Adam
- gradient와 gradient 제곱을 모두 활용하는 방법
$$m_t = \beta_1m_{t-1} + \left(1-\beta_1\right)g_t$$
- $m_t$: momentum
$$v_t = \beta_2v_{t-1} + \left(1-\beta_2\right)g_t^2$$
- $v_t$: EMA of gradient squares
$$W_{t+1} = W_t - \cfrac{\eta}{\sqrt{v_t+\epsilon}}\times\cfrac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}m_t$$
- momentum과 적응형 learning rate 접근법을 조합했다.

## 3) Regularization
- 학습을 방해함으로써 test에서 더 좋은 결과를 얻게 하는 방법

### (1) Early stopping
- test error가 training error와 차이가 벌어지기 전에 일찍 학습을 종료하는 방법

### (2) Parameter Norm Penalty
- loss function에 parameter의 norm을 추가하는 방법
$$total\ cost = loss(\mathcal D; W)+\cfrac{\alpha}{2}\lVert W \rVert_2^2$$
- $\cfrac{\alpha}{2} \lVert W \rVert_2^2$ : parameter norm penalty

### (3) Data Augmetation
- data의 양을 늘림으로써 좋은 결과를 얻을 수 있다.
- 주어진 data의 양이 부족하다면 기존의 데이터를 약간씩 변형하여 사용하는 방법도 있다.
- Mix-up: 임의로 선택된 두 training data의 input과 output을 섞어 데이터를 늘리는 방식
- CutMix: 임의로 선택된 두 training data 중 하나의 일정 영역을 자르고 그 자리를 다른 data로 채우는 방식

### (4) Noise Robustness
- input 또는 weight에 noise를 추가하는 방법
- noise를 추가하면 더 좋은 결과를 얻는다는 실험적 결과가 있다.

### (5) Label Soomthing
- Hard label(one-hot vector)을 soft label(0과 1사이의 수로 표현)로 바꾸는 방법

### (6) Dropout
- 노드에 대한 의존성을 약화시키기 위해 일정 확률로 노드의 값을 0으로 만들어 계산하는 방법

### (7) Batch Normalization
- 각 batch마다 독립적으로 평균과 분산을 계산하는 방법

# 4. Convolutional Neural Networks
## 1) Convolution
- convolution은 kernel을 이용해 주어진 신호를 국소적으로 증폭 또는 감소시킴으로써 필터링하는 것이다.
$$\left( f*g\right)(x) = \sum_{i=-\infty}^\infty f(i)g(x+i) = \sum_{i=-\infty}^\infty f(x+i)g(i)$$
- CNN에서 사용하는 연산은 -가 아니라 +이기 때문에 정확히는 convoluton이 아니라 cross-correlation이지만 관습적으로 convolution이라 부른다.

## 2) 다양한 차원에서의 convolution
- 2D-conv: 
$$\left( f*g\right)(i,\ j) = \sum_p \sum_q f(p,\ q)g(i+p,\ j+q) = \sum_p \sum_q f(i+p,\ j+q)g(p,\ q)$$

- 3D-conv:
$$\left( f*g\right)(i,\ j,\ k) = \sum_p \sum_q \sum_r f(p,\ q,\ r)g(i+p,\ j+q,\ k+r) = \sum_p \sum_q \sum_r f(i+p,\ j+q,\ k+r)g(p,\ q,\ r)$$

- input size $(i,\ j)$, kernel size $(K_i,\ K_j)$로 주어졌을 때 output size $(O_i,\ O_j)$는 $O_i = i-K_i+1,\ O_j = j-K_j+1$이다.
- 채널이 여러개인 2차원 input의 경우 convolution을 채널 개수만큼 적용 (kernel의 채널 수와 input의 채널 수가 같아야 한다.)

- examples
    - I.
    
    $$\begin{bmatrix} a_{11}&a_{12} \\ a_{21}&a_{22} \end{bmatrix} * \begin{bmatrix} b_{11}&b_{12}&b_{13} \\ b_{21}&b_{22}&b_{23} \\ b_{31}&b_{32}&b_{33} \end{bmatrix} = \begin{bmatrix} a_{11}b_{11}+a_{12}b_{12}+a_{21}b_{21}+a_{22}b_{22}&a_{11}b_{12}+a_{12}b_{13}+a_{21}b_{22}+a_{22}b_{23} \\ a_{11}b_{21}+a_{12}b_{22}+a_{21}b_{31}+a_{22}b_{32}&a_{11}b_{22}+a_{12}b_{23}+a_{21}b_{32}+a_{22}b_{33} \end{bmatrix}$$
    $$\quad [2\times2] \qquad \qquad [3\times3] \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \;  [2\times2]$$

    - II.
    $$\left( 5,\ 5,\ 3 \right) * \left( 32,\ 32,\ 3 \right) \rightarrow \left( 28,\ 28,\ 1 \right)$$

    - III.
    $$\mathsf{four\ kernels}\left( 5,\ 5,\ 3 \right) * \left( 32,\ 32,\ 3 \right) \rightarrow \left( 28,\ 28,\ 4 \right)$$

    - IV.
    $$\left( 32,\ 32,\ 3 \right)*\mathsf{four\ kernels}\left( 5,\ 5,\ 3 \right) = \left( 28,\ 28,\ 4 \right) \\\rightarrow \left( 28,\ 28,\ 4 \right)*\mathsf{ten\ kernels}\left( 5,\ 5,\ 4 \right) = \left( 24,\ 24,\ 10 \right)  $$

## 3) Backpropagation of Convolution
다음과 같이 kernel을 w, input을 x, output을 O라 할 때,
$$\begin{bmatrix} w_{11}&w_{12} \\ w_{21}&w_{22} \end{bmatrix} * \begin{bmatrix} x_{11}&x_{12}&x_{13} \\ x_{21}&x_{22}&x_{23} \\ x_{31}&x_{32}&x_{33} \end{bmatrix} = \begin{bmatrix} O_{11}&O_{12} \\ O_{21}&O_{22} \end{bmatrix}$$
$$O_{11} = w_{11}x_{11}+w_{12}x_{12}+w_{21}x_{21}+w_{22}x_{22}$$
$$O_{12} = w_{11}x_{12}+w_{12}x_{13}+w_{21}x_{22}+w_{22}x_{23}$$
$$O_{21} = w_{11}x_{21}+w_{12}x_{22}+w_{21}x_{31}+w_{22}x_{32}$$
$$O_{22} = w_{11}x_{22}+w_{12}x_{23}+w_{21}x_{32}+w_{22}x_{33}$$
이다.

chain rule에 의해 
$$\cfrac{\partial \mathcal{L}}{\partial w_{11}} = \cfrac{\partial \mathcal{L}}{\partial O_{11}}\times\cfrac{\partial O_{11}}{\partial w_{11}} + \cfrac{\partial \mathcal{L}}{\partial O_{12}}\times\cfrac{\partial O_{12}}{\partial w_{11}} + \cfrac{\partial \mathcal{L}}{\partial O_{21}}\times\cfrac{\partial O_{21}}{\partial w_{11}} + \cfrac{\partial \mathcal{L}}{\partial O_{22}}\times\cfrac{\partial O_{22}}{\partial w_{11}}$$
$$= \cfrac{\partial \mathcal{L}}{\partial O_{11}}x_{11} + \cfrac{\partial \mathcal{L}}{\partial O_{12}}x_{12} + \cfrac{\partial \mathcal{L}}{\partial O_{21}}x_{21} + \cfrac{\partial \mathcal{L}}{\partial O_{22}}x_{22}$$
이고, 이것을 matrix 형태로 바꾸면
$$\begin{bmatrix} \cfrac{\partial \mathcal{L}}{\partial w_{11}}&\cfrac{\partial \mathcal{L}}{\partial w_{12}} \\\\ \cfrac{\partial \mathcal{L}}{\partial w_{21}}&\cfrac{\partial \mathcal{L}}{\partial w_{22}} \end{bmatrix} = \begin{bmatrix} x_{11}&x_{12}&x_{13} \\ x_{21}&x_{22}&x_{23} \\ x_{31}&x_{32}&x_{33} \end{bmatrix} * \begin{bmatrix} \cfrac{\partial \mathcal{L}}{\partial O_{11}}&\cfrac{\partial \mathcal{L}}{\partial O_{12}} \\\\ \cfrac{\partial \mathcal{L}}{\partial O_{21}}&\cfrac{\partial \mathcal{L}}{\partial O_{22}} \end{bmatrix}$$
이다.

결국 convolution의 backpropagation은 forward 형태에서 방향만 반대로 convolution되는 방식임을 알 수 있다.

## 4) Convolutional Neural Networks
- CNN은 convolution layer, pooling layer, fully connected layer로 이루어져 있다.
- convolution layer & pooling layer: feature extraction
- fully connected layer: decision making (e.g., classification)

## 5) Convolution Arithmetic
### (1) Stride
- kernel이 움직일 때 지정한 칸 수만큼 건너뛰면서 움직이게 하는 방법
- output의 size를 줄이고 싶을 때 사용한다.

### (2) Padding
- input size가 실제보다 몇 칸 더 큰 것처럼 가정하는 방법
- output size가 너무 줄어드는 것을 원하지 않을 때 사용한다.

### (3) Number of parameters
- kernel size w, input채널 수 C, output 채널 수 D, kernel 개수 K라 하면 parameter 개수는 $w\times w\times C\times D\times K$이다.

## 6) $1\times 1$ Convolution
- 채널 수를 조절하기 위해 사용
- parameter 수를 줄이기 위해 사용
- ReLU와 함께 사용함으로써 비선형성 부여

# 5. Modern CNN
ILSVRC(ImageNet Large-Scale Visual Recognition Challenge)에서 좋은 성과를 보였던 모델들을 살펴본다.

## 1) AlexNet
- Key ideas
    - I. ReLu 사용
    - II. GPI(2 GPUs) 사용
    - III. Local response normalization, Overlapping pooling 사용
    - IV. Data augmetation 사용
    - V. Dropout 사용

- ReLU activation
    - I. 

## 2) VGGNet
## 3) GoogLeNet
## 4) ResNet
## 5) DenseNet

# 6. Computer Vision Applications
# 7. Recurrent Neural Networks
# 8. Transformer 
# 9. Generative Models
