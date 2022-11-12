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
    - Regression Task: 
    $$MSE = \cfrac{1}{N}\sum_{i=1}^N \sum_{d=1}^D \left( y_i^{(d)}-\hat{y}_i^{(d)} \right)^2$$
    - Classification Task: 
    $$CE = -\cfrac{1}{N} \sum_{i=1}^N \sum_{d=1}^D y_i^{(d)}\log\hat{y}_i^{(d)}$$
    - Probabilistic Task: 
    $$MLE = \cfrac{1}{N} \sum_{i=1}^N \sum_{d=1}^D \log\mathcal N \left( y_i^{(d)};\hat{y}_i^{(d)}, 1 \right)$$

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
![mlconcepts_image5](https://user-images.githubusercontent.com/113276742/201312709-60d80a10-93c0-4ef9-a3ec-db20771a0411.png)


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
    
    $$\begin{bmatrix} a_{11}&a_{12} \\\ a_{21}&a_{22} \end{bmatrix} * \begin{bmatrix} b_{11}&b_{12}&b_{13} \\\ b_{21}&b_{22}&b_{23} \\\ b_{31}&b_{32}&b_{33} \end{bmatrix} = \begin{bmatrix} a_{11}b_{11}+a_{12}b_{12}+a_{21}b_{21}+a_{22}b_{22}&a_{11}b_{12}+a_{12}b_{13}+a_{21}b_{22}+a_{22}b_{23} \\\ a_{11}b_{21}+a_{12}b_{22}+a_{21}b_{31}+a_{22}b_{32}&a_{11}b_{22}+a_{12}b_{23}+a_{21}b_{32}+a_{22}b_{33} \end{bmatrix}$$
    
    $$[2\times2] \qquad \qquad [3\times3] \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \quad   [2\times2] \qquad \qquad \qquad \qquad \qquad \qquad \quad$$

    - II.
    $$\left( 5,\ 5,\ 3 \right) * \left( 32,\ 32,\ 3 \right) \rightarrow \left( 28,\ 28,\ 1 \right)$$

    - III.
    $$\mathsf{four\ kernels}\left( 5,\ 5,\ 3 \right) * \left( 32,\ 32,\ 3 \right) \rightarrow \left( 28,\ 28,\ 4 \right)$$

    - IV.
    $$\left( 32,\ 32,\ 3 \right) * \mathsf{four\ kernels} \left( 5,\ 5,\ 3 \right) = \left( 28,\ 28,\ 4 \right) \\ \rightarrow \left( 28,\ 28,\ 4 \right)*\mathsf{ten\ kernels}\left( 5,\ 5,\ 4 \right) = \left( 24,\ 24,\ 10 \right) $$

## 3) Backpropagation of Convolution
다음과 같이 kernel을 w, input을 x, output을 O라 할 때,

$$\begin{bmatrix} w_{11}&w_{12} \\\ w_{21}&w_{22} \end{bmatrix} * \begin{bmatrix} x_{11}&x_{12}&x_{13} \\\ x_{21}&x_{22}&x_{23} \\\ x_{31}&x_{32}&x_{33} \end{bmatrix} = \begin{bmatrix} O_{11}&O_{12} \\\ O_{21}&O_{22} \end{bmatrix}$$

$$O_{11} = w_{11}x_{11}+w_{12}x_{12}+w_{21}x_{21}+w_{22}x_{22}$$
$$O_{12} = w_{11}x_{12}+w_{12}x_{13}+w_{21}x_{22}+w_{22}x_{23}$$
$$O_{21} = w_{11}x_{21}+w_{12}x_{22}+w_{21}x_{31}+w_{22}x_{32}$$
$$O_{22} = w_{11}x_{22}+w_{12}x_{23}+w_{21}x_{32}+w_{22}x_{33}$$
이다.

chain rule에 의해 
$$\cfrac{\partial \mathcal{L}}{\partial w_{11}} = \cfrac{\partial \mathcal{L}}{\partial O_{11}}\times\cfrac{\partial O_{11}}{\partial w_{11}} + \cfrac{\partial \mathcal{L}}{\partial O_{12}}\times\cfrac{\partial O_{12}}{\partial w_{11}} + \cfrac{\partial \mathcal{L}}{\partial O_{21}}\times\cfrac{\partial O_{21}}{\partial w_{11}} + \cfrac{\partial \mathcal{L}}{\partial O_{22}}\times\cfrac{\partial O_{22}}{\partial w_{11}}$$
$$= \cfrac{\partial \mathcal{L}}{\partial O_{11}}x_{11} + \cfrac{\partial \mathcal{L}}{\partial O_{12}}x_{12} + \cfrac{\partial \mathcal{L}}{\partial O_{21}}x_{21} + \cfrac{\partial \mathcal{L}}{\partial O_{22}}x_{22}$$
이고, 이것을 matrix 형태로 바꾸면

$$\begin{bmatrix} \cfrac{\partial \mathcal{L}}{\partial w_{11}}&\cfrac{\partial \mathcal{L}}{\partial w_{12}} \\\ \cfrac{\partial \mathcal{L}}{\partial w_{21}}&\cfrac{\partial \mathcal{L}}{\partial w_{22}} \end{bmatrix} = \begin{bmatrix} x_{11}&x_{12}&x_{13} \\\ x_{21}&x_{22}&x_{23} \\\ x_{31}&x_{32}&x_{33} \end{bmatrix} * \begin{bmatrix} \cfrac{\partial \mathcal{L}}{\partial O_{11}}&\cfrac{\partial \mathcal{L}}{\partial O_{12}} \\\ \cfrac{\partial \mathcal{L}}{\partial O_{21}}&\cfrac{\partial \mathcal{L}}{\partial O_{22}} \end{bmatrix}$$

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
    - I. 선형 모델의 특성 보존
    - II. 경사하강법으로 최적화하기 쉬움
    - III. 좋은 generalization 성능
    - IV. Vanishing gradient problem 극복

## 2) VGGNet
- 3x3 convolution filter로 깊이 증가
    - 3x3 2번과 5x5 1번은 결과는 같지만 parameter 수가 3x3 2번이 더 적다.
- Fully conneted layers를 위한 1x1 convolution 사용
- Dropout 사용
## 3) GoogLeNet
- Network-in-network(NiN)와 inception block을 결합
- Inception block
    - 1x1 kernel을 사용함으로써 채널 수를 줄이고 결과적으로 parameter 수가 줄어듦.

## 4) ResNet
- 깊은 신경망일수록 overfitting이 잘 일어나기때문에 학습이 어렵다.
- Resnet에서는 identity map을 추가
    - $f(x) -> x+f(x)$ weight가 추가된 항과 추가되지 않은 항을 결함
- Parameter 수를 줄이면서 성능은 향상시킴

## 5) DenseNet
- Addition 대신 concatenation 사용
- Dense block
    - 각 layer에서 이전의 layer들을 concat
    - 채널의 수가 기하급수적으로 증가
- Transition Block
    - Batch Norm -> 1x1 Conv -> 2x2 AvgPooling
    - Dimension reduction

## 6) Summary
- Key takeaways
    - VGG: repeated 3x3 blocks
    - GoogLeNet: 1x1 convolution
    - ResNet: skip-connection
    - Densnet: Concatenation

# 6. Computer Vision Applications
## 1) Semantic Segmentation
- Image의 모든 pixel을 labeling하는 문제
- 자율주행에 많이 활용된다.

### (1) Fully Convolutional Network
- 일반적인 CNN 구조는 마지막에 dense layer를 통과시킨다.
- Dense layer를 없애고 그냥 convolution laye를 사용하는 것을 convolutionalization이라고 하고 dense layer가 없어진 network를 fully convolutional network라고 한다.
- Fully connected layers를 fully convolutional layers로 바꾸면 단순 분류만을 알려주는 결과에서 heat map과 같은 형태로 바뀐다.
- FCN은 어떤 input size에서도 실행되지만 output size가 줄어들게 된다.
- 줄어든 output 차원을 다시 늘려줘야 한다.

### (2) Deconvolution (convolution transpose)
- 말그대로 convolution을 반대방향으로 함으로써 output size를 늘려준다.

## 2) Detection
### (1) R-CNN
- I. Image 선택
- II. Image 안에서 특정 영역들 선택
- III. 각 영역에 대해 계산 (Alexnet)
- IV. 계산된 영역들 분류 (SVM)

### (2) SPPNet
- 전반적으로 R-CNN과 비슷하나 선택된 영역들의 tensor만 뽑아서 Image 전체에 대해 CNN을 한번만 거치는 방법

### (3) Fast R-CNN
- I. Image 안에서 영역선택
- II. CNN 사용
- III. ROI pooling을 사용해서 각 영역에 대해 일정 길이의 feature 추출
- IV. 영역 분류와 neural network를 이용해 더 나은 bounding box 조정

### (4) Faster R-CNN
- Faster R-CNN = Regional Proposal Network + Fast R-CNN
- Region Proposal Network : 크기가 미리 정해진 anchor box를 사용하여 bounding box 안에 물체가 있는지를 판단
- RPN은 FCN을 사용하고 총 parameter 수는 $9 \times (4+2)=54$개
    - 9: anchor box size(128, 256, 512) x box ratio(1:1, 1:2, 2:1)
    - 4: box의 x, y 변화량
    - 2: box 분류(물체가 있는지 없는지 판단)

### (5) YOLO
- YOLO(v1): 극도로 빠른 물체 탐색 알고리즘
- Bounding box를 고름과 동시에 분류(bounding box의 선택과 그 box에 대한 결과를 따로 계산하는 faster R-CNN보다 빠르다.)
- Output tensor의 크기는 $S \times S \times (B \times 5 + C)$
    - $S \times S$: box의 cell 개수
    - $B \times 5$: B개의 bounding box와 box의 offset(x, y, w, h)과 confidence
    - C:클래스 개수
     
# 7. Recurrent Neural Networks
## 1) Sequential Model
- 소리, 문자열, 주가 등의 데이터를 sequence data로 분류
- 길이가 정해져있지 않기 때문에 CNN을 적용할 수 없다.
- Sequence data는 독립동등분포 가정을 잘 위배하기 때문에 순서를 바꾸거나 과거 정보에 손실이 발생하면 데이터의 확률분포도 바뀌게 된다.
- I. Naive Sequence Model: 현재 시점 이전의 모든 데이터를 포함
$$p(x_t \vert x_{t-1},\ x_{t-2},\ \cdots)$$
- II. Auto Regressive Model: 현재 시점 이전에서 특정 개수만큼의 데이터만을 포함
$$p(x_t \vert x_{t-1},\ \cdots\ x_{t-\tau})$$
- III. Markov Model: (first-order autoregressive model)
$$p(x_1,\ \cdots,\ x_T) = p(x_T \vert x_{T-1})p(x_{T-1},\vert x_{T-2})\ \cdots p(x_2 \vert x_1)p(x_1) = \prod_{t=1}^T p(x_t \vert x_{t-1})$$
- IV. Latent Autoregressive Model
$$\hat{x} = p(x_t \vert h_t)$$
$$h_t = g(h_{t-1}, x_{t-1})$$
$h_t$: summary of the past

## 2) Recurrent Neural Network
- 잠재변수를 신경망을 통해 반복해서 사용하여 sequence data의 패턴을 학습하는 모델
- Short-term dependencies: 먼 과거의 data는 영향력이 작아진다.
- Vanishing/Exploding gradient
$$h_1 = \phi(W^Th_0 + U^Tx_1)$$
$$h_2 = \phi(W^T\phi(W^Th_0 + U^Tx_1) + U^Tx_2)$$
$$h_3 = \phi(W^T\phi(W^T\phi(W^Th_0 + U^Tx_1) + U^Tx_2) + U^Tx_3)$$
$$h_4 = \phi(W^T\phi(W^T\phi(W^T\phi(W^Th_0 + U^Tx_1) + U^Tx_2) + U^Tx_3) + U^Tx_4)$$
$\phi$가 Sigmoid일 경우 $h_0$는 점점 작아져서 0에 수렴하게 되고(Vanishing), $\phi$가 ReLU일 경우 $h_0$는 점점 커져서(Exploding) 학습이 안 되게 된다.

## 3) Long Short Term Memory
### (1) Forget gate
- 버려야 할 정보를 결정
$$f_t = \sigma(W_f \cdot [h_{t-1},\ x_t] + b_f)$$

### (2) Input gate
- Cell state에 저장해야될 정보를 결정
$$i_t = \sigma(W_i \cdot [h_{t-1},\ x_t] + b_i)$$
$$\tilde{C_t} = tanh(W_c \cdot [h_{t-1},\ x_t] + b_c)$$

### (3) Update gate
- Cell state 업데이트
$$i_t = \sigma(W_i \cdot [h_{t-1},\ x_t] + b_i)$$
$$C_t = f_t * C_{t-1} + i_t*\tilde{C_t}$$

### (4) Output gate
- 업데이트된 cell state를 이용해 output 계산
$$O_t = \sigma(W_o \cdot [h_{t-1},\ x_t] + b_o)$$
$$h_t = O_t * tanh(C_t)$$

## 4) Gated Recurrent Unit(GRU)
- Gate 2개로 LSTM보다 더 간단한 구조(reset gate, update gate)
- No cell state, just hidden state
$$z_t = \sigma(W_z \cdot [h_{t-1},\ x_t])$$
$$r_t = \sigma(W_r \cdot [h_{t-1},\ x_t])$$
$$\tilde{h_t} = tanh(W \cdot [r_t * h_{t-1}, x_t])$$
$$ h_t = (1-z_t) * h_{t-1} + z_t * \tilde{h_t}$$

# 8. Transformer
## 1) Sequential Model
- 무엇이 sequential model을 다루기 어려운 문제로 만드는가?
    - I. Original sequence: 1, 2, 3, 4, 5, 6, 7
    - II. Trimmed sequence: 1, 2, 3, 4, 5
    - III. Omitted sequence: 1, 2, 4, 7
    - IV. Permuted sequence: 2, 3, 4, 6, 5, 7

## 2) Transformer
- Transformer는 sequential data를 처리하고 encoding하는 방법이기 때문에 기계어 번역뿐만 아니라 이미지 분류, 이미지 탐색 등 다양한 분야에서 활용 가능하다.
- Self attention은 RNN과 달리 재귀적으로 작동되지 않고 한번의 작동으로 결과를 내놓는다.
- 동일하지만 공유되지는 않는 6개의 Encoder와 Decoder가 쌓여있는 구조로 되어있다.

### (1) Encoder가 n개의 단어를 처리하는 방법
- 각 Encoder는 self-attention과 feed forward neural network로 구성
- Self-attention
    - n개의 단어가 주어지면 각 단어를 embedding vector로 바꾼다.
    - 각 단어를 embedding할 때 나머지 n-1개를 고려한다. 즉, 각 단어들 사이에 dependency가 존재한다.
    - Embedding vector 외에 query, key, value라는 3가지 벡터를 추가로 만든다. 
    - Score vector를 만든다. 
    $$\mathsf{score} = Q \cdot K^T$$
    - Score vector를 key vector 차원의 제곱근으로 나눠 normalization 한다.
    $$\mathsf{nomalized\ score} = \mathsf{score} \div \sqrt{d_K}$$
    - Normalized score에 softmax를 취해줘서 attention weight를 만든다.
    - 최종적으로 사용할 encoding vector는 value vector에 attention weight를 곱해서 더한 값이다.
    $$\therefore Z = softmax \left( \cfrac{Q \cdot K^T}{\sqrt{d_K}} \right)V$$
- Feed forward neural network에서는 dependency가 없다.
- Multi-headed attention
    - Query, key, value vector를 각각 n개씩 만든다.
    - 위에서 만든 vector로 계산한 encoding vector들을 concat해서 wieght 행렬과의 곱으로 size를 줄인다.
    - 실제 구현된 코드에서는 주어진 input을 전부 사용하지 않고 head 개수만큼 나눠서 따로따로 적용시킨다.
- 문장에서는 단어의 순서 또한 중요하기때문에 positional encoding을 해줘야 하고 방법은 단순히 embedding vector에 positional vector를 더하는 것이다.

### (2) Encoder와 Decoder 사이에 일어나는 일
- Encoder에서 decoder로 key와 value vector를 보낸다.

### (3) Decoder가 m개의 단어를 내놓는 방법
- Self-attention에서는 앞에 있는 단어들만 dependent하게 학습시키는 masking을 사용한다.
- Encoder-Decoder Attention에서는 앞에 있는 단어들의 query vector와 encoder에서 온 key, value vector로 enconding vector를 만든다.
- 마지막으로 단어들의 분포를 만들어서 그 중 가장 적절한 값을 내놓는다.

# 9. Generative Models
## 1) Introduction
### (1) Learning a Generative model
- Suppose that we given images of dogs
- We want to learn a probability distribution $p(x)$ such that
    - Generation: If we sample $\tilde{x} \sim p(x)$, $\hat{x}$ should look like a dog
    - Density estimation: $p(x)$ should be high if $x$ looks like a dog, and low otherwise
        - This is also known as explicit models.
- Then, how can we represent $p(x)$?

### (2) Basic Discrete Distributions
- Bernoulli distribution: (biased) coin flip
    - $D = \{Heads, Tails\}$
    - Specify $P(X=Heads) = p$. Then, $P(X=Tails) = 1-p$
    - write : $X \sim Ber(p)$
- Categorical distribution: (biased) m-sided dice
    - $D = \{1, \cdots, m\}$
    - Specify $P(Y=i) = p_i$ such that 
    $$\sum^m_{i=1}p_i = 1$$
    - write: $Y \sim Cat(p_1, \cdots, p_m)$

### (3) Example
- Modeling a single pixel of an RGB image
    - $(r, g, b) \sim p(R, G, B)$
    - Number of cases?: 256 x 256 x 256
    - How many parameters do we need to specify?
    : 256 x 256 x 256 - 1 (나머지 하나는 자동으로 정해진다.)

## 2) Independence
### (1) Example
- Suppose we have $X_1, \cdots, X_n$ of n binary pixels(a binary image)
    - Number of cases?: $2 \times 2 \times \cdots \times 2 = 2^n$
    - How many parameters do we need to specify? : $2^n -1$

### (2) Structure Through Independence
- What if $X_1, \cdots, X_n$ are independent, then $P(X_1, \cdots, X_n) = P(X_1)P(X_2) \cdots P(X_n)$
    - Number of cases?: $2 \times 2 \times \cdots \times 2 = 2^n$
    - How many parameters do we need to specify?: n
    - $2^n$ entries can be described by just n numbers.

### (3) Conditional Independence
- Three important rules
    - Chain rule: $p(x_1, \cdots, x_n) = p(x_1)p(x_2 \vert x_1)p(x_3 \vert x_1, x_2)\cdots p(x_n \vert x_1, \cdots, x_{n-1})$
    - Bayes' rule: $p(x \vert y) = \cfrac{p(x,y)}{p(y)} = \cfrac{p(y \vert x)p(x)}{p(y)}$
    - Conditional independence: If $x \perp y \vert z$, then $p(x \vert y, z) = p(x \vert z)$
- Using the chain rule, $P(X_1, \cdots, X_n) = P(X_1)P(X_2 \vert X_1)P(X_3 \vert X_1, X_2) \cdots P(X_n \vert X_1, \cdots, X_n)$
- How many parameters?
    - $P(X_1)$: 1 parameter
    - $P(X_2 \vert X_1)$: 2 parameters (One per $P(X_2 \vert X_1 = 0)\ \mathsf{and}\ P(X_2 \vert X_1 = 1)$ )
    - $P(X_3 \vert X_1, X_2)$: 4 parameters
    - Hence, the total number becomes $1+2+2^2+ \cdots +2^{n-1} = 2^n -1$
    - Now, suppose $X_{i+1} \perp X_1, \cdots, X_{i-1} \vert X_i$ (Markov assumption), then $p(x_1, \cdots, x_n) = p(x_1)p(x_2 \vert x_1)p(x_3 \vert x_2) \cdots p(x_n \vert x_{n-1})$
    - How many parameters?: $2n-1$
        - Hence, by leveraging the Markov assumption, we get exponential reduction on the number of parameters.
        - Autoregressive models leverages this conditional independency.

## 3) Autoregressive Models
- Suppose we have 28 x 28 binary pixels.
- Our goal is to learn $P(X) = P(X_1, \cdots, X_784)$ over $X \in \{0, 1\}^{784}$
- Then, how can we parametrize $P(X)$?
    - Let's use the chain rule to factir the joint distribution
    - In other words,
        - $P(X_{1:784}) = P(X_1)P(X_2 \vert X_1)P(X_3 \vert X_2) \cdots$
        - This is called an autoregressive model.
        - Note that we need an ordering (e.g. raster scan order) of all random variables.

### (1) NADE: Neural Autoregressive Density Estimator
- NADE is an explicit model that can compute the density of the given inputs.
- BTW, how can we compute the density of the given image?
    - Suppose that we have a binary image with 784 binary pixels(i.e., $\{x_1, x_2, \cdots, x_{784}\}$)
    - Then, the joint probability is computed by
        - $p(x_1, \cdots, x_{784}) = p(x_1)p(x_2 \vert x_1) \cdots p(x_{784} \vert x_{1:783})$ where each conditional probability $p(x_i \vert x_{1:i-1})$ is computed independently
- In case of modeling continuous random variables, a mixture of Gaussian(MoG) can be used.

### (2) Summary of Autoregressive Model
- Easy to sample from
    - sample $\tilde{x}_0 \sim p(x_0)$
    - sample $\tilde{x}_1 \sim p(x_1 \vert x_0 = \bar{x}_0)$
- Easy to compute probability $p(x=\bar{x})$
    - compute $p(x_0 = \bar{x}_0)$
    - compute $p(x_1 = \bar{x}_1 \vert x_0 = \bar{x}_0)$
    - Multiply together (sum their logarithms)
- Easy to be extended to continuous variables. For example, we can choose mixture of Gaussians.

## 4) Maximum Likelihood Learning
- Given a training set of examples, we can cast the generative model learning process as finding the best-approximating density model from the model family.
- Then, how can we evaluate the goodness of the approximation?
- KL-divergence
$$D(P_{data} \Vert P_\theta) = E_{X \sim P_{data}}\left[log\left( \cfrac{P_{data}(x)}{P_\theta(x)} \right)\right] = \sum_x P_{data}(x)log \cfrac{P_{data}(x)}{P_\theta (x)}$$
- We can simplify this with
$$D(P_{data} \Vert P_\theta) = E_{X \sim P_{data}}\left[log\left( \cfrac{P_{data}(x)}{P_\theta(x)} \right)\right] = E_{X\sim P_{data}}\left[ logP_{data}(x) \right] - E_{X \sim P_{data}}\left[logP_\theta(x)\right]$$
- As the first term does not depend on $P_\theta$, minimizing the KL-divergence is equivalent to maximizing the expected log-likelihood.
$$arg \min_{P_\theta} D(P_{data} \Vert P_\theta) = arg \min_{P_\theta} - E_{X \sim P_{data}}\left[log P_\theta(x)\right] = arg \max_{P_\theta} E_{x \sim P_{data}}\left[ log P_\theta(x)\right]$$
- Approximate the expected log-likelihood
$$ E_{x \sim P_{data}}\left[logP_\theta(x)\right]$$
with the empirical log-likelihood
$$E_D\left[logP_\theta(x)\right] = \cfrac{1}{|D|}\sum_{x \in D}logP_\theta(x)$$
- Maximum likelihood learning is then:
$$\max_{P_\theta}\cfrac{1}{|D|}\sum_{x\in D}logP_\theta(x)$$
- Variance of Monte Carlo estimate is high:
$$V_P[\tilde{g}] = V_P \left[\cfrac{1}{T}\sum^T_{t=1}g(x^t)\right] = \cfrac{V_P[g(x)]}{T}$$
- For maximum likelihood learning, empirical risk minimization(ERM) is often used.
- However, ERM often suffers from its overfitting.
    - Extreme case: The model remembers all training data.
    
    $$ p(x) = \frac{1}{|D|}\sum^{|D|}_{i=1}\delta(x, x_i)$$
- To achieve better generalization, we typically restrict the hypothesis space of distributions that we search over.
- However, it could deteriorate the performance of the generative model.
- Usually, MLL if prone to under-fitting as we often use simple parametric such as spherical Gaussians.
- What about other ways of measuring the similarity?
    - KL-divergence leads to maximum likelihood learning or Variational Autoencoder(VAE).
    - Jensen-Shannon divergence leads to Generative Adversarial Network(GAN).
    - Wasserstein distance leads to Wasserstein Autoencoder(WAE) or Adversarial Autoencoder(AAE).

## 5) Latent Variable Models
- Is an autoencoder a generaive Model? No!

### (1) Variational Autoencoder
- The objective is simple: Maximize $P_\theta(x)$
- Variational inference(VI)
    - The goal of VI is to optimize the variational distribution that best matches the posterior distribution.
        - Posterior distribution: $P_\theta(z|x)$
        - Variational distribution: $q_\theta(z|x)$
    - In particular, we want to find the variational distribution that minimizes the KL divergence between the true posterior.
    $$\underset{Maximum\ Likelihood\ Learning \uparrow} {logP_\theta(x)} = \int_z q_\phi(z|x)logP_\theta(x)dx$$
    $$= E_{z \sim q_\phi}(z|x)\left[ log\cfrac{P_\theta(x) P_\theta(z|x)}{P_\theta(z|x)}\right] = E_{z \sim q_\phi}(z|x)\left[log \cfrac{P_\theta(x,z)}{P_\theta(z|x)}\right]$$
    $$= E_{z \sim q_\phi}(z|x)\left[log \cfrac{P_\theta(x)q_\phi(z|x)}{P_\theta(z|x)q_\phi(z|x)}\right]$$
    $$= E_{z \sim q_\phi}(z|x)\left[log \cfrac{P_\theta(x,z)}{q_\phi(z|x)}\right] + E_{z \sim q_\phi}(z|x)\left[log \cfrac{q_\phi(z|x)}{P_\theta(z|x)}\right]$$
    $$= E_{z \sim q_\phi}(z|x)\left[log \cfrac{P_\theta(x,z)}{q_\phi(z|x)}\right] + D_{KL}(q_\phi(z|x)\Vert P_\theta(z|x))$$

    ### (2) Evidence Lower Bound
    $$\underset{ELBO\uparrow}{E_{z \sim q_\phi}(z|x)\left[log \cfrac{P_\theta(x,z)}{q_\phi(z|x)}\right]} = \int log\cfrac{P_\theta(x|z)P(z)}{q_\phi(z|x)}q_\phi(z|x)dz$$
    $$=E_{q_\phi(z|x)}[P_\theta(x|z)] - D_{KL}(q_\phi(z|x) \Vert p(z))$$
    - $E_{q_\phi(z|x)}[P_\theta(x|z)]$: Reconstruction Term. This term minimizes the reconstruction loss of an auto-encoder.
    - $D_{KL}(q_\phi(z|x) \Vert p(z))$: Prior Fitting Term. This term enforces the latent distribution to be similar to the prior distribution.
    - Key Limitation
        - It is an intractable model(hard to evaluate likelihood)
        - The prior fitting terms should be differentiable, hence it is hard to use diverse latent prior distributions.
        - In most cases, we use an isotropic Gaussian where we have a closed-form for the prior fitting term.
        $$D_{KL}(q_\phi(z|x)\Vert \mathcal{N}(0,I)) = \frac{1}{2}\sum^D_{i=1}(\sigma^2_{z_i} + \mu^2_{z_i} - log(\sigma^2_{z_i}) -1)$$

## 6) Generative Adversarial Networks
$$\min_G \max_D V(D,G) = E_{x\sim P_{data}(x)}[logD(x)] + E_{x\sim P_z(z)}[log(1-D(G(z)))]$$

### (1) GAN Objective
- GAN is a two player minimax game between generator and discriminator.
    - Discriminator objective
    $$\max_D V(G,D) = E_{x\sim P_{data}}[logD(x)] + E_{x\sim P_G}[log(1-D(x))]$$
    - The optimal discriminator is 
    
    $$D^*_G = \cfrac{P_{data}(x)}{P_{data}(x) + P_G(x)}$$
    - Generator objective
    $$\min_G V(G,D) = E_{x\sim P_{data}}[logD(x)] + E_{x\sim P_G}[log(1-D(x))]$$
    - Plugging in the optimal discriminator, we get
    
    $$V(G, D^*_G(x)) = E_{x\sim P_{data}}\left[log\cfrac{P_{data}(x)}{P_{data}(x)+P_G(x)}\right] + E_{x\sim P_G}\left[log\cfrac{P_G(x)}{P_{data}(x)+P_G(x)}\right]$$
    
    $$= E_{x\sim P_{data}}\left[log\cfrac{P_{data}(x)}{\cfrac{P_{data}(x)+P_G(x)}{2}}\right] + E_{x\sim P_G}\left[log\cfrac{P_G(x)}{\cfrac{P_{data}(x)+P_G(x)}{2}}\right] - log4$$
    
    $$= D_{KL}\left[P_{data}, \cfrac{P_{data}+P_G}{2}\right] + D_{KL}\left[P_G, \cfrac{P_{data}+P_G}{2}\right] - log4$$
    
    $$= 2D_{JSD}[P_{data}, P_G] - log4$$

## 7) Diffusion Models
- Diffusion models progressively generate images from noise
- Forward (diffusion) process progressively injects noise to an image.
$$P_\theta(X_{t-1}|X_t): = \mathcal{N}(X_{t-1};\mu_\theta(X_t,t), \sum_\theta(X_t,t))$$
- The reverse process is learned in such a way to denoise the perturbed image back to a clean image.
