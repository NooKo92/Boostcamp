# 1. Introduction to PyTorch
- 딥러닝을 할 때 코드를 처음부터 전부 짜는 대신 PyTorch를 사용한다.
- PyTorch = Numpy + AutoGrad + Function

# 2. torch.nn.Module (Custom Model)
- 모든 신경망 module들을 위한 기본 class
- module은 다른 module을 포함할 수 있다.

## 1) torch.nn.Sequential
- 여러개의 module들을 순차적으로 실행시키는 class

        import torch
        from torch import nn

        class Add(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, x):
                return x + self.value


        calculator = nn.Sequential(Add(4), Add(3), Add(6)) # y = x + 4 + 3 + 6
        x = torch.tensor([2])
        output = calculator(x)
        # output = 15

## 2) torch.nn.parameter.Parameter
- module의 parameter로 여겨지는 일종의 tensor
- tensor와 달리 자동저장 된다.
- output tensor에 gradient를 계산하는 함수인 grad_fn가 생성된다.

        import torch
        from torch import nn
        from torch.nn.parameter import Parameter

        class Linear(nn.Module):
            def __init__(self):
                super().__init__()

                self.W = Parameter(torch.Tensor([[0,1,2],[2,1,0]]))
                self.b = Parameter(torch.Tensor([[1,1,1],[1,1,1]]))

            def forward(self, x):
                output = torch.addmm(self.b, x, self.W) # self.b + x @ self.W

                return output

        x = torch.Tensor([[1, 2], [3, 4]])

        linear = Linear()
        output = linear(x)
        # output = [[5,4,3], [9,8,7]]

## 3) torch.nn.Module.apply
- 특정 함수를 전체가 아닌 부분 module에 적용시키는 명령어

        import torch
        from torch import nn

        @torch.no_grad()
        def init_weights(m):
            print(m)
            if type(m) == nn.Linear:
                m.weight.fill_(1.0)
                print(m.weight)

        net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        net.apply(init_weights)

# 3. PyTorch Dataset & DataLoader
- 데이터가 모델에 들어가는 것은 다음과 같은 과정으로 이루어진다.
$$\mathsf{Data} \Rightarrow \mathsf{Dataset} \Rightarrow \mathsf{DataLoader} \Rightarrow \mathsf{Model}$$

## 1) Dataset
- torch.utils.data의 Dataset class를 상속해서 만들고 map-style과 iterable-style의 두가지 타입이 있다.
- map-style dataset은 __init__, __len__, __getitem__ 의 세가지 메서드로 구성된다.

        from torch.utils.data import Dataset

        class CustomDataset(Dataset):
            def __init__(self,):
                pass

            def __len__(self):
                pass

            def __getitem__(self, idx):
                pass

### (1) __init__ 메서드
- 데이터의 위치나 파일명 등의 초기화 작업
- 모든 데이터를 메모리에 로드하지 않고 효율적으로 작업 가능

### (2) __len__ 메서드
- Dataset의 최대 요소 수를 반환

### (3) __getitem__ 메서드
- Dataset의 특정 데이터를 반환

## 2) DataLoader
- 모델 학습을 위해 데이터를 mini batch 단위로 제공한다.

        DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
                   batch_sampler=None, num_workers=0, collate_fn=None,
                   pin_memory=False, drop_last=False, timeout=0,
                   worker_init_fn=None)

### (1) sampler / batch_sampler
- 데이터의 index를 원하는 방식으로 조정
- shuffle이 False여야 한다.
- SequentialSampler : 항상 같은 순서
- RandomSampler : 랜덤, replacemetn 여부 선택 가능, 개수 선택 가능
- SubsetRandomSampler : 랜덤 리스트, 위와 두 조건 불가능
- WeigthRandomSampler : 가중치에 따른 확률
- BatchSampler : batch단위로 sampling 가능
- DistributedSampler : 분산처리 (torch.nn.parallel.DistributedDataParallel과 함께 사용)

### (2) num_workers
- 데이터를 불러올 때 사용하는 sub-process 개수
- num_workers를 너무 높이면 CPU와 GPU 사이에 교류가 많아져 병목이 생길 수 있다.

### (3) collate_fn
- sample list를 batch 단위로 바꾸는 기능
- zero-padding이나 Variable Size 데이터 등 데이터 사이즈를 맞추기 위해 많이 사용한다.

### (4) pin_memory
- tensor를 CUDA 고정 메모리에 할당시킨다.

### (5) drop_last
- 마지막 batch를 제외시킨다.
- batch 단위로 데이터를 불러온다면 마지막 batch의 크기가 다른 경우가 발생할 수 있다. 이 때, batch 크기 의존도가 높은 함수를 사용하는 상황이라면 마지막 batch를 제외하는 것이 좋은 선택이 될 수 있다.

# 4. 일반적인 학습 과정

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader 
    from network import CustomNet
    from dataset import ExampleDataset
    from loss import ExampleLoss

    ###################
    # Custom modeling #
    ###################

    # 모델 생성
    model = CustomNet()
    model.train()

    # 옵티마이저 정의
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Example(params, lr=lr)

    # 손실함수 정의
    loss_fn = ExampleLoss()

    ###############################
    # Custom Dataset & DataLoader #
    ###############################

    # 학습을 위한 데이터셋 생성
    dataset_example = ExampleDataset()

    # 학습을 위한 데이터로더 생성
    dataloader_example = DataLoader(dataset_example)

    ##############################################
    # Transfer Learning & Hyper Parameter Tuning # 
    ##############################################
    for e in range(epochs):
        for X,y in dataloader_example:
            output = model(X)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()