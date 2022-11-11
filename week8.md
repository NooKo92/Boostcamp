# 1. MLOps 개론
## 1) 모델 개발 프로세스
### (1) Research vs Product
||Research|Product|
|---|---|---|
|단계|문제정의 -> EDA -> Feature Engineering -> Train -> Predict |-> Deploy |
|데이터|고정(static)|계속 변함(Dynamic-Shifting)|
|중요 요소|모델 성능(Accuracy, RMSE 등)|모델 성능, 빠른 Inference 속도, 해석 가능함|
|도전 과제|더 좋은 성능을 내는 모델(SOTA), 새로운 구조의 모델|안정적인 운영, 전체 시스템 구조|
|학습|(데이터는 고정)모델 구조, 파라미터 기반 재학습|시간의 흐름에 따라 데이터가 변경되어 재학습|
|목적|논문 출판|서비스에서 문제 해결|
|표현|Offline|Online|

### (2) MLOps란?
- $\mathsf{MLOps} = \mathsf{ML(Machine Learning)}\ +\ \mathsf{Ops(Operations)}$
- ML 모델을 운영하면서 반복적으로 필요한 업무를 자동화시키는 과정
- ML Engineering + Data Engineering + Cloud + Infra
- 아직까지 best 라이브러리가 없기 때문에 tool에 집중하지 말고 방법에 집중할 것

## 2) MLOps Component
### (1) Infra
- 클라우드: AWS, GCP, Azure, NCP 등
- 온 프레미스: 회사나 대학원의 전산실에 서버를 직접 설치

### (2) Serving
- Batch Serving: 일정 주기마다 output 전달
- Online Serving: 실시간으로 output 전달

### (3) Experiment, Model Management
- 모델을 만들고 개선시키는 과정을 기록하는 것

### (4) Feature Store
- ML에 반복적으로 사용되는 feature를 미리 저장해두는 것

### (5) Data Validation
- Production의 feature가 Research의 feature와 얼마나 다른지 feature 분포를 확인하는 것

### (6) Continuous Training
- 모델을 다시 학습시키는 것
    - I. 새로운 데이터가 있는 경우
    - II. 일정 주기마다
    - III. 결과(Metric)가 좋지 않은 경우
    - IV. 요청시

### (7) Monitoring
- 모델 또는 인프라의 성능 지표를 기록하는 것

### (8) AutoML
- 자동으로 모델을 만드는 기능

### (9) 정리
- 모든 요소가 항상 존재해야 하는 것은 아님
- MLOps를 처음부터 진행하는 것이 오히려 비효율적일 수 있음

# 2. Product Serving
## 1) Model Serving
- Serving: ML모델을 웹, 앱에서 사용할 수 있게 만드는 행위
- 서비스화라고 표현할 수도 있음
- 크게 Online/Batch Serving으로 구분
- Serving vs Inference
    - Serving: 모델을 서비스화하는 관점
    - Inference: 모델을 사용하는 관점
    - Serving-Inference 용어가 혼재되어 사용되기도 함

## 2) Online Serving
### (1) Web Server Basic
- Web Server: HTTP를 통해 웹 브라우저에서 요청하는 HTML 문서나 오브젝트를 전송해주는 서비스 프로그램 (from. Wikipedia)
- ML 모델 서버: 어떤 데이터를 제공하며 예측해달라고 요청하면 모델을 사용해 예측 값을 반환하는 서버
- 즉, 서버란 요청하면 반환하는 것

### (2) API
- Application Programming Interface: 운영체제나 프로그래밍 언어가 제공하는 기능을 제어할 수 있게 만든 인터페이스
- Pandas, PyTorch 등의 라이브러리도 API의 한 종류

### (3) Online Serving Basic
- 요청이 올 때마다 실시간으로 예측
- 단일 데이터 예제: 기계 고장 예측, 음식 배달 소요 시간 예측
- 전처리 서버, ML모델 서버, 서비스 서버를 분리시킬 수도 있고 합칠 수도 있다.
- Online Serving 구현 방식
    - I. 직접 API 웹 서버 개발
    - II. 클라우드 서비스 활용
    - III. Serving 라이브러리 활용
- Serving 할 때 Python 버전, 패키지 버전 등 Dependency가 매우 중요
- Latency(지연 시간)에 영향을 미치는 요소
    - I. Input 데이터를 기반으로 Database에 있는 데이터를 추출해서 예측해야 하는 경우
    - II. 모델이 수행하는 연산
    - III. 결과 값에 대한 보정이 필요한 경우

## 3) Batch Serving
- 주기적인 학습 또는 예측
- 관련 라이브러리는 따로 없음
- Online Serving보다 구현이 수월하며 Latency 문제가 없음
- 실시간으로 활용할 수 없기 때문에 cold start 문제 존재

## 4) Online Serving vs Batch Serving
- Input 관점
    - 데이터를 하나씩 요청: Online
    - 여러 데이터를 한꺼번에 처리: Batch
- Output 관점:
    - API 형태로 결과를 바로 반환해야 하는 경우: Online
    - 서버와 통신이 필요한 경우: Online
    - 실시간으로 처리하지 않아도 되는 경우: Batch
- Batch Serving으로 시작하여 점점 API형태로 변환하는 것이 좋음

# 3. ML Project Life Cycle
## 1) ML Project Flow
### (1) 문제 정의의 중요성
- 문제정의: 특정 현상을 파악하고 그 현상에 있는 문제를 정의하는 과정
- How보다 Why에 집중하자

### (2) 구체적인 문제 정의
- 앞선 현상을 더 구체적이고 명확한 용어로 정리해보기
- 데이터를 수집해 문제의 원인과 해결방안 고민해보기
- 시간은 한정적이기 때문에 간단한 방법부터 사용해보기

### (3) 프로젝트 설계
- ML Project 과정
    - I. 문제정의
    - II. 최적화할 Metric 선택
    - III. 데이터 수집, 레이블 확인
    - IV. 모델 개발
    - V. 예측 오차 분석
    - VI. 모델 학습 -> 데이터 수집 -> 모델 학습...
    - VII. 모델 배포
    - VIII. Metric에 문제 있으면 Metric 수정
    - IX. II부터 다시 시작
- 문제정의에 기반해서 프로젝트 설계
    - I. 문제 구체화
    - II. ML 문제 타당성 확인 (비즈니스적 관점, 패턴 유무, 목적함수 설계 가능 여부, 복잡성, 데이터 존재 여부, 일의 반복성)
    - III. 목표 설정, 지표 설정(Goal: 최종목표, Objective: 세부 단계 목표)
        - Objective가 여러개인 경우 분리하는 것이 좋음
        - 모델이 재학습하지 않도록 모델을 분리
    - IV. 제약 조건 확인 (일정, 예산, 관련된 사람, privacy, 기술적 제약, 윤리적 이슈, 성능)
    - V. 베이스 라인, 프로토타입 제작
        - 모델이 더 좋아졌다고 판단할 수 있는 Baseline이 필요
        - 꼭 모델일 필요는 없고 자신이 모델이라 생각하면서 Rule Base 규칙 설계
    - VI. 평가방법 설계 (작게는 모델의 성능 지표;RMSE 일 수도 있고 크게는 비즈니스 지표; 매출 일 수 있음)
### (4) Action
- 모델 개발 후 배포 & 모니터링
- 앞서 정의한 지표가 어떻게 변하는지 파악하기

## 2) 비즈니스 모델
- I. 회사의 비즈니스 파악하기: 회사가 어떤 서비스, 가치를 제공하고 있는가?
- II. 데이터를 활용할 수 있는 부분은 어디인가? (input)
- III. 모델을 활용한다고 하면 예측의 결과가 어떻게 활용되는가? (output)

# 4. Notebook 베이스 - Voila
## 1) Voila
- Voila의 장점
    - I. Jupyter Notebook 결과를 쉽게 웹 형태로 띄울 수 있음
    - II. Ipywidget, Ipyleaflet 등 사용가능
    - III. Jupyter Notebook의 Extension 있음 (=Notebook에서 바로 대시보드로 변환 가능)
    - IV. Python, Julia, C++ 코드 지원
    - V. 고유한 템플릿 생성 가능
    - VI. 너무 쉬운 러닝 커브
- Voila 사용 시 Tip
    - Voila는 유저별로 새로운 Notebook kernel을 실행시키는 구조
    - Voila는 노트북을 사용하지 않을 때 자동으로 종료해야 함
    - Jupyter Notebook의 config에서 cull 옵션 확인
    - 아무 설정을 하지 않을 경우 하나의 cell이 30초 이상 진행되면 Timeout error 발생
    - Jupyter Notebook의 password를 사용해 암호를 지정할 수 있음

## 2) Ipywidget
- Voila를 Ipywidget과 같이 사용하면 interactive 효과를 줄 수 있음
- Ipywidget도 Notebook 프로젝트

# 5. 웹 서비스 형태 - streamlit
## 1) Streamlit
- 다른 조직(프론트 엔드/ PM 조직)의 도움없이 빠르게 웹 서비스를 만들게 하기 위한 것
- Voila, streamlit의 순서는 프로젝트의 요구 조건에 따라 다름!

## 2) Streamlit의 대안
- I. R의 shiny
- II. Flask, Fast API: 백 엔드를 직접 구성 + 프론트 엔드 작업도 진행
- III. Dash: 제일 기능이 풍부한 python 대시보드 라이브러리
- IV. Voila: Jupyter Notebook을 바로 시각화 가능

## 3) Streamlit의 장점
- I. Python 스크립트 코드를 조금만 수정하면 웹을 띄울 수 있음
- II. 백 엔드 개발이나 HTTP 요청을 구현하지 않아도 됨
- III. 다양한 component를 제공해 대시보드 UI 구성할 수 있음
- IV. Streamlit Cloud도 존재해서 쉽게 배포 가능
- V. 화면 녹화 기능 존재

## 4) Session state
- Code가 수정되거나 사용자가 streamlit의 위젯과 상호작용하면 전체 streamlit 코드가 다시 실행된다. (Data Flow)
- Streamlit의 Data Flow로 인해 중복 이벤트를 할 수 없어 개발된 것이 session_state
- Global variable처럼 서로 공유 할 수 있는 변수가 필요할 때 사용

## 5) @st.cache
- 캐싱: 성능을 위해 메모리 등에 저장하는 행위
- 데이터도 매번 다시 읽는 것을 막기 위해 @st.cache 데코레이터로 캐싱하면 좋음

# 6. Linux & Shell Command
## 1) Linux
### (1) Linux를 알아야 하는 이유
- I. 무료, 오픈소스
- II. 안전성, 신뢰성
- III. Shell 커맨드, Shell 스크립트 사용가능

### (2) CLI, GUI
- CLI: Command Line Interface (Terminal)
- GUI: Graphic User Interface (Desktop)

### (3) 대표적인 Linux 배포판
- I. Debian: 온라인 커뮤니티에서 제작해 배포
- II. Ubuntu: 영국의 캐노니컬이라는 회사에서 배포. 초보자들이 쉽게 접근할 수 있고 설치가 편함
- III. RedhatL 레드햇이라는 회사에서 배포
- IV. CentOS: Redhat의 브랜드와 로고를 제거한 버전

## 2) Shell Command
### (1) Shell의 종류
- Shell: 사용자가 문자를 입려해 컴퓨터에 명령할 수 있도록 하는 프로그램
- 터미널/콘솔: Shell읠 실행하기 위해 문자 입력을 받아 컴퓨터에 전달. 프로그램의 출력을 화면에 작성
- sh: 최초의 shell
- bash: Linux 표준 shell
- zsh: Mac 카탈리나 OS 기본 shell

### (2) Shell을 사용하는 경우
- I. 서버에서 접속해서 사용하는 경우
- II. Crontab 등 Linux의 내장 기능을 활용하는 경우
- III. 데이터 전처리
- IV. Docker를 사용하는 경우
- V. 수백대의 서버를 관리할 경우
- VI. Jupyter Notebook의 Cell 앞에 !를 붙이면 shell 커맨드가 사용됨
- VII. 터미널에서 python3, jupyter notebook도 shell 커맨드
- VIII. Test code 실행
- IX. 배포 파이프라인 실행

### (3) Redirection & Pipe
- Unix에서 동작하는 프로그램은 커맨드 실행 시 3개의 stream이 생성됨
    - stdin: 입력, 0으로 표현
    - stdout: 출력, 1로 표현
    - stderr: 디버깅 정보나 에러 출력, 2로 표현
- Redirection: 프로그램의 출력(stdout)을 다른 파일이나 스트림으로 전달 (>, >>)
- Pipe: 프로그램의 출력(stdout)을 다른 프로그램의 입력으로 사용 (|)

### (4) Shell 스크립트
- .sh 파일을 생성하고, 그 안에 shell 커맨드를 추가
- Python처럼 if, while, case문이 존재하며 작성 시 bashname.sh로 실행 가능
- Shell 스키립트 = shell 커맨드의 조합

# 7. Docker
## 1) 가상화란?
- 특정 소프트웨어 환경을 만들고 local, production 서버에세 그대로 활용

## 2) Docker 등장 전
- 가상화 기술로 주로VM(Virtual Machine)을 사용
- VM은 호스트 머신이라고 하는 실제 물리적인 컴퓨터위에, OS를 포함한 가상화 소프트웨어를 두는 방식 (ex. Windows에서 Linux 실행)
- OS 위에 OS를 하나 더 실행시킨다는 점에서 VM은 리소스를 굉장히 많이 사용하고 이런 경우를 "무겁다"라고 표현
- Container: VM의 무거움을 크게 덜어주면서, 가상화를 좀 더 경량화된 프로세스의 개념으로 만든 기술

## 3) Docker 소개
- Docker: Container 기술을 쉽게 사용할 수 있도록 나온 도구
- Docker Image: 컨테이너를 실행할 때 사용할 수 있는 "템플릿" (Read only)
- Docker Container: Docker Image를 활용해 실행된 인스턴스 (write 가능)

## 4) Docker로 할 수 있는 일
- 다른 사람이 만든 소프트웨어를 가져와서 바로 사용할 수 있음
- 자신만의 이미지를 만들면 다른 사람에게 공유할 수 있음

# 8. MLflow
## 1) MLflow 개념 잡기
### (1) MLflow란?
- ML 실험, 배포를 쉽게 관리할 수 있는 오픈소스
- MLflow가 해결하려고 했던 pain point
    - I. 실험을 추적하기 어렵다.
    - II. 코드를 재현하기 어렵다.
    - III. 모델을 패키징하고 배포하는 방법이 어렵다.
    - IV. 모델을 관리하기 위한 중앙 저장소가 없다.

### (2) MLflow의 핵심 기능
- I. Experiment Management & Training
- II. Model registry
- III. Model serving

### (3) MLflow Component
- I. MLflow Tracking: 결과를 local, server에 기록해 여러 실행과 비교 가능
- II. MLflow Project: ML 프로젝트 코드를 패키징하기 위한 표준
- III. MLflow Model: 모델 파일과 코드저장
- IV. MLflow Registry: 중앙 모델 저장소

## 2) MLflow 서버로 배포하기
- MLflow Architecture

|Python code<br>(with MLflow package)|Tracking Server<br>(localhost: 5000)|Artifact store<br>(File)|
|:---:|:---:|:---:|
|- 모델을 만들고 학습하는 코드<br>- mlflow run으로 실행|- python 코드가 실행되는 동안 parameter, metric, model 등 메타 정보 저장<br>- 파일 혹은 DB에 저장|- python 코드가 실행되는 동안 생기는 model file, image 등의 아티팩트를 저장<br>- 파일 혹은 스토리지에 저장|

- I. mlflow run (python code)
- II. 기록 요청 (python code -> tracking server)
- III. 기록 (tracking server -> DB)
- IV. 아티팩트 저장 (python code -> artifact store)