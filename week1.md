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
- $\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta} \approx \mathbf{y}$ (y의 추정치는 값에 근사한다.)
## 1) 경사하강법으로 선형회귀 계수 $\boldsymbol{\beta}$ 구하기
- $\mathbf{y} - \hat{\mathbf{y}} = \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rightarrow 0$ 이 되도록 하는 $\boldsymbol{\beta}$를 구하는 것이 목적이다.
- 이를 위해서는 $\mathbf{y} - \mathbf{X}\boldsymbol{\beta}$ 의 gradient를 구해야 한다.
$$\nabla_\beta\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2 = \left( \partial_{\beta_1}\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2, \cdots,  \partial_{\beta_d}\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2 \right)$$
$$\Rightarrow \partial_{\beta_k}\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2 = \partial_{\beta_k} \left\\{ \frac{1}{n}\sum_{i=1}^n \left( y_i - \sum_{j=1}^d X_{i,j}\beta_j \right)^2 \right\\}^{1/2}$$
$$= \left( \partial_{\beta_k} \left\\{ \cfrac{1}{n}\sum_{i=1}^n\left( y_i - \sum_{j=1}^d X_{i,j}\beta_j \right)^2 \right\\}\right) / \left( {2\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2}\right) $$

 
