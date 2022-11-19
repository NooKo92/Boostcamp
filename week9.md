# 1. DKT 이해
## 1) DKT Task 이해
- Deep Knowledge Tracing: DL을 이용해서 하는 지식 상태 추적
- 지식 상태는 계속 변화하기 때문에 지속적으로 추적해야 한다.

## 2) Metric 이해
- ACC: Accuracy
- AUC: Area Under The Roc Curve

### (1) Confusion Matrix
|-|-|Predicted||-|
|---|---|---|---|---|
|||Negative|Positive||
|Actual|Negative|True Negative|False Positive|Specicity $=\cfrac{TN}{TN+FP}$|
||Positive|False Negative|True Positive|Recall, Sensitivity, True Positive Rate(TPR) $=\cfrac{TP}{TP+FN}$|
|||Accuracy $=\cfrac{TP+TN}{TP+TN+FP+FN}$|Precision, Positive Predictive Value(PPV) $=\cfrac{TP}{TP+FP}$|F1-score $=2 \times \cfrac{Recall \times Precision}{Recall + Precision}$|

### (2) AUC
- Roc curve 밑부분 면적이 넓을수록 모형 성능이 높다.
- Rules of Thumbs
    - I. Poor model(0.5 ~ 0.7)
    - II. Fair model(0.7 ~ 0.8)
    - III. Good model(0.8 ~ 0.9)
    - IV. Excellent model(0.9 ~ 1.0)
- x축은 False Positive Rate(FPR) $= \cfrac{FP}{TN+FP}$
- y축은 True Positive Rate(TPR) $= \cfrac{TP}{TP+FN}$
- 맞으면 위로 틀리면 오른쪽으로 움직인다.
- AUC는 척도 불변과 분류 임계값 불변이라는 두 가지 특징이 있다. 이러한 두 가지 특징은 특정 사용 사례에서 AUC의 유용성을 제한할 수 있다는 단점이 있다.

## 3) DKT History 및 Trend
- History
    - I. ML(1994)
    - II. DL(2015)
    - III. Transformer(2019) - 대회 우승권에 많았음
    - IV. GNN(2019)
- NLP 분야의 영향을 많이 받아왔음.
    - I. Model
    - II. Data
    - III. Regularization term
    - IV. Embedding

## 4) Sequence Data
- Sequence 모델
    - I. RNN: 장문장에서 학습이 어려움
    - II. LSTM: 장기 기억과 단기 기억 모두 사용
    - III. SEQ2SEQ: 문장이 길어졌을 때 문제 발생
    - IV. Attention: 학습 속도가 느림
    - V. Transformer: 병렬처리로 학습 속도 높이고 위치 정보로 어순 정렬

# 2. DKT EDA
## 1) i-Scream 데이터 분석
### (1) 기본적인 내용 파악
- 대분류를 하는 feature engineering에 정답은 없다. 성능이 잘 나오기만 하면 된다.

### (2) 기술 통계량 분석
- 평균, 중앙값, 최대/최소 등을 뽑고 시각화하는 작업을 거친다.
- 사용자 분석
    - 사용자 당 푼 문항 수
    - 사용자 별 정답률
- 문항별/시험지 별 정답률 분석
    - 평균, 중앙값, 최대/최소

### (3) 일반적인 EDA
- 문항을 더 많이 푼 학생이 문제를 더 잘 맞추는가?
- 더 많이 노출된 태그가 정답률이 더 높은가?
- 문항을 풀수록 실력이 늘어나는가?
- 문항을 푸는데 걸린 시간과 정답률 사이의 관계는?
- 그 밖에 생각해 볼 수 있는 것들
    - 더 많이 노출된 시험지는 정답률이 높을까?
    - 같은 시험지의 내용이나 같은 태그의 내용을 연달아 풀면, 정답률이 오를까?

# 3. Baseline Model
## 1) Sequence 모델링
- Non-Sequential Data: 행 하나 당 사람 한 명의 정보가 있음(Titanic)
- Sequential Data: 한 사람에 대해 여러 행에 걸쳐 정보가 있음(Transaction)
    - I. 집계(aggregation), Feature Engineering
    - II. Transaction 그대로 사용 + Feature Engineering

## 2) Tabular Approach
### (1) Feature Engineering
- I. 문제를 푼 시점에서의 사용자의 정답률(이전까지의 정답 수/이전까지의 풀이 수)
- II. 문제 및 시험별 난이도(전체 정답 수/전체 풀이 수)

### (2) Train/Valid Data Split
- 사용자 단위로 split을 해야 유저의 실력이 보존된다.

### (3) Model Training
- 하이퍼 파라미터 및 Feature들을 조절해가며 최고의 모델을 확인

## 3) Sequential Approach
### (1) Sequential Data
- I. One-to-One
- II. One-to-Many
- III. Many-to-One
- IV. Many-to-Many
- V. Sequence-to-Sequence

### (2) LSTM & Transformer
- LSTM or Transformer를 활용한 주식가격 예측
    - I. 종가만 사용
    - II. 시가, 종가, 거래량 사용(모두 연속형 변수)
    - III. 시가, 종가, 거래량, 섹터 사용(연속형과 범주형의 조합)

### (3) Embedding
||||||
|-|-|-|-|:-:|
|categorical<br>(-1, seq_len, cate_size)|-> Embedding<br>(-1, seq_len, cate_size, emb_size)|-> Reshape<br>(-1, seq_len, cate_size $\times$ emb_size)|-> Linear<br>(-1, seq_len, hidden_size//2)|-> Layernorm|
|||||$\downarrow$<br>Concat<br>(-1, seq_len, hidden_size)<br>$\uparrow$|
|continuous<br>(-1, seq_len, cate_size)|-|-|->Linear<br>(-1, seq_len, hidden_size//2)|->Layernorm|

### (4) Input Transformer
- 사용자 단위로 Sequence를 생성
- DKT의 경우 Sequence가 짧은 user에 대해 padding을 앞에 추가

### (5) FE and Model
- I. Make ground baseline with no feature engineering
- II. Find good cross-validation strategy(리더보드와 cv결과의 방향성이 일치하는가?)
- III. Feature Selection
- IV. Tune Model(crude tuning)
- V. Try other Models(Never forget about NN)
- VI. Try Blending/Stackin/Ensemble
- VII. Final tuning

# 4. Transformer Architecture 설계
## 1) Data Science Bowl
### (1) 대회 개요
- input: 영유아들의 기초 수학 개념학습을 위한 교육에서의 모든 학습 과정
- output: 평가 문제를 맞힐 것인지 예측

### (2) How to embed features for Transformer
- 서로 다른 범주형/연속형 데이터들을 어떻게 임베딩했는가?
- BERT를 어떻게 활용했는가?

## 2) Riiid!
### (1) 대회 개요
- input: 토익 공부를 한 학생들의 학습과정
- output: 마지막에 푼 문항을 맞힐 것인지 예측

### (2) Using Half-sequence
- 임베딩 된 2개의 sequence를 하나로 이어붙여 sequence의 길이를 반으로 줄이는 대신, 하나의 임베딩 차원을 2배로 늘려 학습시킴

## 3) Predict Molecular Properties
### (1) 대회 개요
- input: 분자 내 원자 간 결합 정보, 원자 간 가림막 효과 등
- output: 원자 간 결합상수
- Sequence 내에서 위치가 중요하지 않음

### (2) How to use Transformer for non-sequence
- Transformer가 Sequence에 유리한 이유
    - I. Sequencce 안에서 모든 token이 다른 token을 참조한다.
    - II. Positional Embedding을 추가하여, sequence 내에서 위치정보까지 반영할 수 있다.
- Positional Embedding을 제거하면 원자쌍의 순서가 반영되지 않고 학습이 가능

## 4) Mechanics of Actions(MoA)
### (1) 대회 개요
- input: 약물 종류, 투여량, 투여한 시간 등
- output: 207개의 화학반응 중 어떤 것들이 일어날지 예측

### (2) Transformer, always?
- Transformer가 잘 작동하지 않음
    - I. Sequence로 묶을 수 있는 데이터가 없다.
    - II. 너무 많은 Feature
    - III. 예측 해야 할 class 수에 비해 적은 데이터
- 이 대회에서는 1D-CNN이 좋은 성능을 보였음

# 5. Kaggle Riid Competiotion winner's solution 탐색
## 1) Feature Engineering
### (1) Bottom-up vs Top-down
- Bottom-up: Data-driven
- Top-down: Hypothesis-driven

### (2) Numerical vs Categorical
|Numerical|공통|Categorical|
|-|-|-|
|- min, max, 1Q, median, 3Q, range<br>- mean, std<br>- kurtosis(참조), skew(왜도)<br>- Target(0 or 1)에 따른 분포|- 파일 크기<br>- column수, row수<br>- 중복 row|- Total Unique count, percent<br>- Value별 Unique count, percent<br>- Value 별 count, percent<br>- 최빈도 값|

### (3) 사용자 별 문항을 푸는 패턴
- I. 문항의 정답을 하나로 찍는 경우
- II. 사용자가 문항을 푸는데 걸린 평균 시간보다 오래 걸렸을 경우
- III. 이전에 풀었던 문항이 다시 등장하는 경우
- IV. 사용자가 연속으로 정답을 맞히고 있는 경우

### (4) 문항별 특징
- I. 문항, 시험지, 태그의 평균 정답률
- II. 문항, 시험지, 태그의 빈도
- III. Data Leakage
- IV. 문항 고유의 Feature 뽑아내기: MF, ELO, IRT

## 2) Winner's solution - Last Query Transformer RNN
### (1) Resolving deficits
- 많은 대회에서 사용되는 기법들
    - I. LGBM, DNN
        - 아주 많은 Feature를 만들어내야 하고, 유의미한 것을 찾기도 어려움
    - II. Transformer
        - Sequence 길이의 제곱에 비례한 시간 복잡도를 가지기 때문에 사용하기 부담스러움
- 두 가지 문제를 모두 해결한 유저가 1등을 차지함

### (2) Small number of Features
- 다수의 Feature를 사용하지 않음
    - 5개의 Feature만 사용(LGBM을 사용하는 모델은 70~80개 이상의 Feature를 사용하기도 함)
    - Feature Engineering에 시간을 줄이면 모델에 다양한 시도를 해볼 수 있는 시간이 늘어난다.

### (3) Advantages of Last Query ONLY
- 마지막 Query만 사용하여 시간복잡도를 낮춤
    - Transformer의 Attention score의 시간복잡도는 $O(L^2d)$이다.
    - 여기서 마지막 Query만 사용한다면 $O(Ld)$로 줄어든다.

### (4) Adding LSTM
- 문제간 특징을 Transformer로 파악하고 sequence 사이 특징들을 LSTM을 활용해 뽑아낸 뒤 마지막에 DNN을 통해 sequence 별 정답을 예측함.

# 6. DKT with RecSys
- DKT와 RecSys 모두 0과 1 사이에서 결과를 예측한다는 점에서 비슷하다.

## 1) Task: DKT + RecSys
- DKT: w/ Transformer(Encoder only, Encoder+Decoder), w/o Transformer(Sequential, Non-Sequential)
- RecSys: w/ GNN(Sequential, Non-Sequential), w/o GNN(Seq., Non-Seq.)
- DKT with RecSys - Graph Neural Network
    - GNN은 둘 사이에 link가 생길지를 예측하는 방법
    - 문제와 유저 사이에 link가 생기면 문제를 푸는 것이고 안 생기면 못 푸는 것이다.

## 2) Graph Neural Network
### (1) Graph structure
- Node = Vertex: user, item
- Edge = Link
- Graph $G = (V, E)$
- 그래프란 노드와 그 노드를 연결하는 간선을 하나로 모은 구조
- Edge: directed/undirected (방향 유무), weighted/unweighted(연결 유무)

### (2) Representation Learning
- Goal: Efficient node Embedding
- input(d-dimension vector) -> output($d_l$-dimension vector)

### (3) Procedural Components of GNN
- Data -> Graph construction -> GNN -> Optimization(pair-wise / point-wise loss) -> backpropagation -> GNN -> ...

### (4) Tasks on Graphs
- I. Node Classification: $softmax(z_n)$
- II. Graph Classification: $softmax(\sum_n z_n)$
- III. Link Prediction: $p(A_{ij}) = \sigma(z^T_iz_j)$

### (5) Matrix Representation of Graph
- I. Adjacency matrix: Undirected graph
    - N by N
    - symmetric
- II. Adjacency matrix: Directed graph
    - N by N
    - asymmetric
- III. Adjacency matrix: Directed graph(+ self loop)
    - Adjacency matrix + Identity matrix
- IV. Adjacency matrix: weighted directed graph
    - Edge information
    - $\alpha_{ij} \neq \alpha_{ji},\ i \neq j$
- V. Node-feature matrix: node의 정보를 vector로 만들고 그것을 쌓아서 만든 matrix
- VI. Degree matrix: 각 노드마다 몇 개의 노드와 연결되어 있는 지를 나타내는 diagonal matrix
- VII. Laplacian matrix = Degree matrix - Adjacency matrix

## 3) Graph Convolution Networks
### (1) Convolution on Graphs
- 이미지를 인접한 픽셀끼리 연결되어 있는 그래프라고 한다면, Graph Convolution의 한 종류로 볼 수 있음
- Graph Convolution은 노드와 연결된 이웃들의 정보를 weighted avg. 한다

### (2) Update Calculation
- Self loop가 있는 node1에 node2, 3, 4번이 연결되어 있는 상황을 가정하면
$$H^{(l+1)}_1 = \sigma\left(H^{(l)}_1W^{(l)} + H^{(l)}_2W^{(l)} + H^{(l)}_3W^{(l)} + H^{(l)}_4W^{(l)} + b^{(l)} \right)$$

$$\Rightarrow H_i^{(l+1)} = \sigma\left(\sum_{j\in \mathcal{N}{(i)}} H_j^{(l)} W^{(l)} + b^{(l)}\right) = \sigma\left(AH^{(l)}W^{(l)} + b^{(l)} \right)$$
- GCN의 목적: 주변 노드의 정보를 통해 자신의 정보를 hidden representation으로 만들 수 있는 필터(weight)를 찾는 것
$$f\left(H^{(l)}, A \right) = \sigma\left(AH^{(l)}W^{(l)} \right)$$
- 가중치(W)의 특징
    - I. Sharing
    - II. Order invariant
    - III. Independent number of node
- GCN의 weight는 전체 노드 정보를 고려한 차원 변환 파라미터
    - 그래프 구조적 정보가 반영된 파라미터 -> spectral
    - 새로운 노드가 오면 전체 그래프와의 연결성을 고려 -> Transductive

## 4) Graph Attention Networks
### (1) Convolution Weight & Attention Coefficient
$$H^{(l+1)} = \sigma\left(\sum_{j\in \mathcal{N}(i)}\alpha^{(l)}_{ij}H^{(l)}_jW^{(l)} \right)$$
$$\alpha_{ij} = softmax(e_{ij}) = \cfrac{e_{ij}}{\mathsf{exp}\left(\sum_{k\in \mathcal{N}(i)}e_{ik} \right)}\ ,\ e_{ij} = LeakyReLU\left(\alpha^T\left[H_iW, H_jW \right] \right)$$

### (2) Multi-head Attention
- Average

$$H_i^{(l+1)} = \sigma\left(\frac{1}{k}\sum_{k=1}^k\sum_{j\in \mathcal{N}(i)}\alpha^{(l)}_{ij}H^{(l)}_jW^{(l)} \right)$$
- Concatenation

$$H_i^{(l+1)} = \bigcup_{k=1}^K\sigma\left(\sum_{j\in \mathcal{N}(i)}\alpha^{(l)}_{ij}H^{(l)}_jW^{(l)} \right)$$

### (3) Attention
- 각 node 간 벡터를 참고하여 연산을 할 때, 전체를 동일한 비율로 참고하는 것이 아니라, 연관이 있는 node에 좀 더 집중

### (4) Over-smoothing
- 각 node마다 한 번씩 계산을 수행하고 한번 더 하게되면 node의 이웃의 이웃의 정보까지 얻게 된다.
- 이것이 계속되면 모든 node의 정보가 희석되는 over-smoothing이 발생한다.
- GNN에서 Layer가 5개이상부터는 over-smoothing이 발생된다고 한다.

# 7. GNN-RecSys
## 1) Neural Graph Collaborative Filtering(NGCF)
### (1) Introduction
- user-item interaction을 embedding 함수에 넣어야 한다.

### (2) 1-hop layer
- Message aggregation

$$e_u^{(l)} = LeakyReLU\left(m_{u \leftarrow u} + \sum_{i\in N_u} m_{u \leftarrow i} \right)$$
- Message construction

$$m_{u\leftarrow i} = \cfrac{1}{\sqrt{\left|\mathcal{N_u}\right| \left|\mathcal{N_i}\right|}}\left(W_1e_i + W_2(e_i \odot  e_u) \right)$$
$$m_{u\leftarrow u} = W_1e_u$$
- Embedding layer

$$E = \left[e_{u_1}, \cdots, e_{u_N}, e_{i_1}, \cdots, e_{i_M} \right]$$

### (3) NGCF Architecture
- Embeddings -> Embedding Propagation Layers (N-hop Layers) -> Prediction Layer 

$$\hat{y}_ {NGCF}(u, i) = {e_u^\ast}^T,\ e_i^\ast,\quad Loss = \sum_{(u,i,j) \in O} \ln\sigma\left(\hat{y}_ {ui} - \hat{y}_ {uj} \right)(\mathsf{BPR\ Loss})$$

## 2) LightGCN
- NGCF에서 feature transformation과 non-linear activation function을 둘 다 제거하면 성능 향상됨
- Message propagation

$$e_u = \sum_{i \in \mathcal{N}_ u}\cfrac{1}{\sqrt{N_u}\sqrt{N_i}}e_i,\ e_i = \sum_ {i\in\mathcal{N}_u}\cfrac{1}{\sqrt{N_u}\sqrt{N_i}}e_u$$

