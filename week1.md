# 1. 경사하강법(Gradient Descent)
$y = f(x)$라는 함수에서 변수 x에 미분값을 빼준 값을 갱신해 나가면 x, y는 극소값에 도달하게 된다.
## 1) 변수가 하나인 경우
$y = x^2 + 5x + 3$ 이라는 함수의 경사하강법 알고리즘은 다음과 같다.

    import sympy as sym
    import numpy as np
    from sympy.abc import x
  
    def func(val):
      fun = sym.poly(x**2 + 5*x + 3)
      return fun.subs(x, val), fun #함수값과 함수식
    
    def func_gradient(fun, val):
      function = fun(val)[1]
      diff = sym.diff(function, x) #function을 x에 대해 미분하는 코드
      return diff.subs(x, val), diff #미분값과 미분된 식
    
    def gradient_descent(fun, init_point, lr_rate=1e-2, epsilon=1e-5):
      cnt = 0
      val = init_point
      diff = func_gradient(func, val)[0]
      while np.abs(diff) > epsilon:
        val = val - lr_rate*diff
        diff = func_gradient(func, val)[0]
        cnt += 1
        
      print('함수: {}\n연산횟수: {}\n최소점: ({}, {})'.format(func(val)[1], cnt, val, func(val)[0]))
    
    gradient_descent(fun=func, init_point=3)
    #함수: poly(x**2 + 5*x + 3, x, domain='zz') 
    #연산횟수: 689
    #최소점: (-2.49999504402803, -3.24999999997544)
    
## 2) 변수가 벡터인 경우
$z = x^2 + 2x + y^2 + 3y$ 의 경사하강법 알고리즘은 다음과 같다.

    def func(val):
        fun = sym.poly(x**2 + 2*x + y**2 + 3*y)
        return fun.subs([(x, val[0]), (y, val[1])]), fun #함수값과 함수식

    def func_gradient(fun, val):
        z = fun(val)[1]
        diff_x = sym.diff(z, x) #function을 x에 대해 편미분
        diff_y = sym.diff(z, y) #function을 y에 대해 편미분
        diff = [diff_x, diff_y]
        diff_val = [diff_x.subs([(x, val[0]), (y, val[1])]), diff_y.subs([(x, val[0]), (y, val[1])])]
        return diff_val, diff #미분값과 미분된 식

    def gradient_descent(fun, init_point, lr_rate=1e-2, epsilon=1e-5):
        cnt = 0
        val = init_point
        grad = func_gradient(func, val)[0]
        while max(grad) > epsilon:
            val = [val[i] - lr_rate*grad[i] for i in range(len(val))]
            grad = func_gradient(func, val)[0]
            cnt += 1
        
        print('함수: {}\n연산횟수: {}\n최소점: ({}, {})'.format(func(val)[1], cnt, val, func(val)[0]))
    
    gradient_descent(fun=func, init_point=(0,0))
    #함수: Poly(x**2 + 2*x + y**2 + 3*y, x, y, domain='ZZ')
    #연산횟수: 625
    #최소점: ([-0.999996716800237, -1.49999507520035], -3.24999999996497)
    
# 2. 선형회귀(Linear Regression)
- 독립변수 x와 종속변수 y의 선형상관관계를 추정하는 방법
- $\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta} \approx \mathbf{y}$ (y의 추정치는 y값에 근사한다.)
## 1) 경사하강법으로 선형회귀 계수 $\boldsymbol{\beta}$ 구하기
- $\mathbf{y} - \hat{\mathbf{y}} = \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rightarrow 0$ 이 되도록 하는 $\boldsymbol{\beta}$를 구하는 것이 목적이다.
- 이를 위해서는 $\mathbf{y} - \mathbf{X}\boldsymbol{\beta}$ 의 gradient를 구해야 한다.
$$\nabla_\beta\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2 = \left( \partial_{\beta_1}\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2, \cdots,  \partial_{\beta_d}\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2 \right)$$
$$\Rightarrow \partial_{\beta_k}\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2 = \partial_{\beta_k} \left\{ \frac{1}{n}\sum_{i=1}^n \left( y_i - \sum_{j=1}^d X_{i,j}\beta_j \right)^2 \right\}^{1/2}$$
$$= \left( \partial_{\beta_k} \left\{ \cfrac{1}{n}\sum_{i=1}^n\left( y_i - \sum_{j=1}^d X_{i,j}\beta_j \right)^2 \right\}\right) / \left( {2\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2}\right) $$
$$= \left( \partial_{\beta_k} \left[ \cfrac{1}{n}\sum_{i=1}^n\left\{ y_i^2 - 2y_i\sum_{j=1}^d X_{i,j}\beta_j + \left( \sum_{j=1}^d X_{i,j}\beta_j \right)^2 \right\} \right]\right) / \left( {2\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2}\right) $$
$$= \left\{ \cfrac{1}{n}\sum_{i=1}^n\left( - 2y_i X_{i,k} + 2X_{i,k}\sum_{j=1}^d X_{i,j}\beta_j \right) \right\} / \left( {2\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2}\right) $$
$$= \left\{ -\cfrac{2}{n}\sum_{i=1}^n X_{i,k} \left( y_i - \sum_{j=1}^d X_{i,j}\beta_j \right) \right\} / \left( {2\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2}\right) $$
$$= -\cfrac{ \mathbf{X_k^T}\left( \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right)}{n\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2} $$

$$ \therefore \nabla_\beta\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2 = -\cfrac{\mathbf{X^T} \left( \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right)}{n\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2} $$

- $\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2$ 대신 $\left( \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2 \right)^2$을 사용하면 식이 더 간단해진다.

$$ \partial_{\beta_k} \left( \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2 \right)^2 = \partial_{\beta_k} \left\{ \frac{1}{n}\sum_{i=1}^n \left( y_i - \sum_{j=1}^d X_{i,j}\beta_j \right)^2 \right\} $$

$$= -\cfrac{2}{n}\mathbf{X_k^T} \left( \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right) $$

$$ \therefore \nabla_\beta\left( \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2 \right)^2 = -\cfrac{2}{n}\mathbf{X^T}\left( \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \right)$$

## 2) 경사하강법 기반 선형회귀 알고리즘
$y = 2x + 3$ 의 데이터가 주어졌을 때, 이를 $y = wx + b$로 가정하고 계수 w와 b를 찾기 위한 경사하강법 기반 선형회귀 알고리즘은 다음과 같다.

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.figure as fig

    np.random.seed(0)
    x = np.random.randn(10000,1)*2
    y = 2*x + 3 #실제 y값
    x = np.concatenate((x, np.ones_like(x)), axis=1)
    w, b = 0.0, 0.0 #계수 w와 b 초기값
    beta = np.array([w,b]).reshape(2,1) #계수 행렬 beta
    lr_rate = 1e-5
    errors = []

    for t in range(100):
        y_hat = x @ beta #y 추정값
        error = y - y_hat
        grad = -np.transpose(x) @ error #norm2 제곱의 gradient
        beta -= lr_rate * grad
        errors.append(np.linalg.norm(error))

    print('w: {}\nb: {}\nerror: {}'.format(beta[0,0], beta[1,0], errors[-1]))
    #w: 1.9999989742674702 
    #b: 2.9999192350146244
    #error: 0.008972042876632457

    # 그래프를 그리기 위한 코드
    plt.figure(figsize=(20,5))
    ax = plt.plot(errors)
    plt.xlabel('trial')
    plt.ylabel('error')

    plt.show()

## 3) 확률적 경사하강법(stochastic gradient descent)
- 전체 데이터에서 일부만을 사용하는 경사하강법
- 데이터의 일부만을 사용하기 때문에 좀 더 효율적이다. (연산량이 mini-batch size / total size 만큼 감소) 
- 위와 같은 문제에서 확률적 경사하강법을 이용한 알고리즘은 다음과 같다.
    
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.figure as fig
    
        np.random.seed(0)
        x = np.random.randn(10000,1)*2
        y = 2*x + 3 #실제 y값
        x = np.concatenate((x, np.ones_like(x)), axis=1)
        w, b = 0.0, 0.0 #계수 w와 b 초기값
        beta = np.array([w,b]).reshape(2,1) #계수 행렬 beta
        lr_rate = 0.005
        errors = []

        for t in range(100):
            idx = np.random.choice(1000,10) # 크기 10짜리 mini-batch
            m_x = x[idx]
            m_y = y[idx]
        
            y_hat = m_x @ beta #y 추정값
            error = m_y - y_hat
            grad = -np.transpose(m_x) @ error #norm2 제곱의 gradient
            beta -= lr_rate * grad
            errors.append(np.linalg.norm(error))

        print('w: {}\nb: {}\nerror: {}'.format(beta[0,0], beta[1,0], errors[-1]))
        #w: 2.000747075329951 
        #b: 2.9830003462765644
        #error: 0.056048243120771266

        # 그래프를 그리기 위한 코드
        plt.figure(figsize=(20,5))
        ax = plt.plot(errors)
        plt.xlabel('trial')
        plt.ylabel('error')

        plt.show()
        
- 위 두 알고리즘의 그래프 비교
![1](https://user-images.githubusercontent.com/113276742/194876091-fb3b270a-f1cd-44e8-9889-e0ad7e1d5674.png)
![2](https://user-images.githubusercontent.com/113276742/194876122-64235aaf-3d7b-4277-b088-49ff9495330c.png)

# 3. 딥러닝의 학습 방법
## 1) 신경망 수식
$$ \begin{bmatrix} -\mathbf{O_1}- \\
-\mathbf{O_2}- \\
\vdots \\
-\mathbf{O_n}- \end{bmatrix} = \begin{bmatrix} -\mathbf{X_1}- \\
-\mathbf{X_2}- \\
\vdots \\
-\mathbf{X_n}- \end{bmatrix} \begin{bmatrix} \mathbf{w_{11}} \mathbf{w_{12}} \cdots \mathbf{w_{1p}} \\
\mathbf{w_{21}} \mathbf{w_{22}} \cdots \mathbf{w_{2p}} \\
\vdots \\
\mathbf{w_{d1}} \mathbf{w_{d2}} \cdots \mathbf{w_{dp}} \end{bmatrix}+ \begin{bmatrix} \vdots \\
\mathbf{b_1} \quad \mathbf{b_2} \cdots \mathbf{b_p} \\
\vdots \end{bmatrix}$$

$$\ \left(n \times p \right) \qquad\ \ \left(n \times d \right) \qquad \quad \left(d \times p \right) \qquad \qquad \qquad \left(n \times p \right) \qquad $$
- 각 행벡터 $\mathbf{O_i}$는 데이터 $\mathbf{X_i}$와 가중치 행렬 $\mathbf{w}$사이의 행렬곱과 절편 $\mathbf{b}$벡터의 합으로 표현된다고 가정한다.

## 2) 활성함수(Activation Function)
- 활성함수 $\sigma$는 비선형 함수로 잠재벡터 $z = (z,\ \cdots,\ z_q)$의 각 노드에 개별적으로 적용되어 새로운 잠재벡터 $H = \left( \sigma(z_1),\ \cdots,\ \sigma(z_q) \right)$를 만든다.

### (1) Sigmoid 함수
<p align="center"><img src="https://user-images.githubusercontent.com/113276742/196037460-7b091159-f934-4cc9-8603-5d9d6ba939f9.png"></p>

$$\sigma(x) = \cfrac{1}{1+e^{-x}}$$

### (2) Hyperbolic Tangent 함수
<p align="center"><img src="https://user-images.githubusercontent.com/113276742/196037039-da1a4c33-0652-4977-8f67-9bc8d9f37454.png"></p>
$$\tanh(x) = \cfrac{e^x-e^{-x}}{e^x+e^{-x}}$$

### (3) Rectified Linear Unit 함수
<p align="center"><img src="https://user-images.githubusercontent.com/113276742/196037043-be6e8b2c-e4bc-4eb8-b7ff-3084f26c2100.png"></p>
$$ReLU(x) = max(0,x)$$

## 3) 다층 인공 신경망(Multi-Layer Perceptron, MLP)
- 신경망을 여러층 쌓으면 MLP라고 한다.
- hidden layer를 사용함으로써 비선형성을, 목적함수를 근사하는데 필요한 노드의 숫자를 줄임으로써 효율성을 얻을 수 있기 때문에 MLP를 사용한다.
$$\mathbf{O} = \mathbf{Z}^{(L)}$$
$$\uparrow$$
$$\mathbf{H}^{(l)} = \sigma\left(\mathbf{Z}^{(l)}\right)$$
$$\mathbf{Z}^{(l)} = \mathbf{H}^{(l-1)}\mathbf{W}^{(l)} + \mathbf{b}^{(l)}$$
$$\uparrow$$
$$\mathbf{H}^{(1)} = \sigma\left(\mathbf{Z}^{(1)}\right)$$
$$\mathbf{Z}^{(1)} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}$$

## 4) 역전파(Backpropagation)
- 손실함수 $\mathcal{L}$을 최소화하는 $\mathbf{w}^{(l)}$을 찾기위해 $\partial\mathcal{L}/\partial\mathbf{w}^{(l)}$을 계산하는 방법
$$\cfrac{\partial\mathcal{L}}{\partial\mathbf{w}^{(l)}} = \cfrac{\partial\mathcal{L}}{\partial\mathbf{O}} \times\ \cdots\ \times \cfrac{\partial\mathbf{Z}^{(l+1)}}{\partial\mathbf{H}^{(l)}} \times \cfrac{\partial\mathbf{H}^{(l)}}{\partial\mathbf{Z}^{(l)}} \times \cfrac{\partial\mathbf{Z}^{(l)}}{\partial\mathbf{w}^{(l)}}$$
