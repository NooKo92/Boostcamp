# 1. 경사하강법
y = f(x)라는 함수에서 변수 x에 미분값을 빼준 값을 갱신해 나가면 x, y는 극소값에 도달하게 된다.
## 1) 변수가 하나인 경우
y = x^2 + 5x + 3 이라는 함수의 경사하강법 알고리즘은 다음과 같다.

    import sympy as sym
    import numpy as np
    from sympy.abc import x
  
    def func(val):
      fun = sym.poly(x**2 + 5*x + 3)
      return fun.subs(x, val), fun #함수값과 함수식
    
    def func_gradient(fun, val):
      _, function = fun(val)
      diff = sym.diff(function, x) #function을 x에 대해 미분하는 코드
      return diff.subs(x, val), diff #미분값과 미분된 식
    
    def gradient_descent(fun, init_point, lr_rate=1e-2, epsilon=1e-5):
      cnt = 0
      val = init_point
      diff, _ = func_gradient(func, val)
      while np.abs(diff) > epsilon:
        val = val - lr_rate*diff
        diff, _ = func_gradient(func, val)
        cnt += 1
        
      print('함수: {}, 연산횟수: {}, 최소점: ({}, {})'.format(func(val)[1], cnt, val, func(val)[0]))
    
    gradient_descent(fun=func, init_point=3)
    #함수: poly(x**2 + 5*x + 3, x, domain='zz'), 연산횟수: 689, 최소점: (-2.49999504402803, -3.24999999997544)
    
## 2) 변수가 벡터인 경우
