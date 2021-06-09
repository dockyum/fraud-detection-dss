# Fraud Detection (with SOCAR data)

## 0. 요약
```
1) 프로젝트 목적
    - 자동차 보험 사기 탐지 모델을 통해 보험 사기 가능성이 있는 사고를 필터링
    - 보험 사기를 판단하기 위한 실사 대상 건수를 줄이는 효과가 있을 것으로 기대
2) 프로젝트 결과
    - 가장 적합한 모델은 RandomForest로 recall 0.71, accuracy 0.81, precision 0.009, auc 0.77 (test score기준)
        -> 실제 활용을 위해선 precision의 향상 필요
```

## 1. 소개
### 1) 주제
- 자동차 보험 사기 탐지 모델링
### 2) 기대효과
- 보험 사기일 가능성이 있는 건들을 사전에 필터링, 실사가 필요한 총 대상 건수 감소에 따른 실사 효율성 향상
### 3) 데이터셋
- SOCAR로부터 제공받았으며, 데이터 구조 등 데이터 관련 정보들은 일체 비공개    


## 2. 진행 프로세스
- 데이터 전처리(2️⃣) -> 하이퍼 파라미터 튜닝(3️⃣) -> 모델 평가(4️⃣) 
- 위 단계를 반복하여 나온 평가 점수표를 통해 최종 모델 선정(5️⃣)

    ![process_overall](https://user-images.githubusercontent.com/78459305/117934187-5a0b5080-b33d-11eb-8b37-91f8622102b9.png)


## 3. 각 프로세스별 상세설명
### 1) raw data 수집
- SOCAR로부터 제공받은 raw data를 사용

### 2) 데이터 전처리 
- 전처리 적용 유무의 효과 비교를 위해 
    #### (1) 틀린 데이터 (wrong data)
    - feature들 간에 논리적으로 앞뒤가 맞지 않는 데이터 일부 존재
    - 논리적으로 합당하지 않은 데이터이므로 이상치(outlier)가 아닌 틀린(wrong) 데이터라고 판단, 모든 과정에서 항상 제거

    #### (2) Null data
    - 제공 받은 데이터는 null값이 `0`등 다른 값으로 대체된 상태
    - null data가 과반수 이상인 feature들의 경우, feature를 유지 또는 삭제, 총 2가지 경우에 대해 진행
    #### (3) Sampling
    - 비정상(사기) 라벨 데이터 수가 절대적으로 적어 undersampling은 데이터셋 특성 상 부적합하다고 판단, oversampling만 진행
    - 총 5가지 방법 (SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, RandomOverSampling)으로 샘플링 진행
    #### (4) One-hot Encoding
    - 범주 데이터의 경우, 원핫인코딩을 적용 또는 미적용, 총 2가지 경우에 대해 진행
    #### (5) Scaling
    - 연속형 데이터의 경우, feature간 단위 차이에 따른 학습 방해를 예방하기 위해 스케일링 진행
    - 총 4가지 방법 (MinMax, Standard, Robust, None(미적용))으로 스케일링 진행  

### 3) 알고리즘 & 하이퍼파라미터 튜닝
- 총 5가지 모델 사용, cv를 통해 각 모델에서 최적의 파라미터 도출 

    #### (1) Logistic Regression
    - class_weight:  (none, {0:01, 1:1.0}, {0:0.005, 1:1}, 'balanced')
    #### (2) Support Vector Classification
    - C:  (0.1, 1.0)
    - class_weight:  (none, {0:01, 1:1.0}, {0:0.005, 1:1}, 'balanced')
    #### (3) Light GBM
    - n_estimators:  (50, 100, 200, 400) 
    - num_leaves:  (4, 8, 16) 
    - class_weight:  (none, {0:01, 1:1.0}, {0:0.005, 1:1}, 'balanced') 
    #### (4) Decision Tree
    - max_depth:  (3, 4, 6, 8, 10, 30) 
    - max_features:  (none, sqrt, log2)
    - class_weight:  (none, {0:01, 1:1.0}, {0:0.005, 1:1}, 'balanced')
    #### (5) Random Forest
    - max_depth:  (4, 6, 8, 10, 30)
    - n_estimators:  (50, 100, 200, 400)
    - class_weight:  (none, {0:01, 1:1.0}, {0:0.005, 1:1}, 'balanced')


### 4) 모델 성능 평가
- 상기 전처리 유무 x 알고리즘 조합을 통해 총 800가지 경우의 모델 생성
- 성능 평가의 지표로 recall을 메인 지표로 사용
    - '사기' 탐지를 최대한 많이 하기 위해 recall을 메인 지표로 사용


### 5) 최적 모델 선정
- recall이 가장 높은 모델은 test recall이 0.85까지 나왔으나 precision이 0.006으로 미비 \
-> 최적 모델이 아니라고 판단

- 실사 대상 건수를 낮춘다는 목적에 부합하게 `test auc` 기준으로 최적 모델 선정

![score table](https://user-images.githubusercontent.com/18084336/121317784-94204000-c945-11eb-87b8-a538449c8d4b.png)

## 4. 결론
- Null값이 과반수인 열 삭제, 원핫인코딩 미적용, 스케일러 미적용, SVMSMOTE로 샘플링, Logistic Regression을 사용하였으며 class_weight를 {0: 0.005, 1: 1}으로 적용한 RandomForest를 최적의 모델로 평가
- test recall이 **0.7142**, precision은 **0.0087**로 나타남


## 5. 추후 보완점
- GridSearchCV 과정에서 `scoring='recall'`로 설정하고 진행했는데, 때문에 모든 모델에서 train recall 점수는 높고 다른 지표는 낮았다. 추후 scoring 파라미터에 따른 결과 비교 필요
- 상대적으로 데이터 전처리를 통한 성능 개선 시도는 부족했으며, 추후 진행 시 null값 처리 등 데이터 전처리에 더 초점을 두어 진행 예정


## 6. 멤버 & 수행업무
#### [김도겸](https://github.com/dockyum)
  * 폴더구조 관리
  * 데이터 프로세싱 : EDA, 파이프라인 코드 작성 및 관리, SVMSMOTE,ROS 샘플러 활용한 CV 진행
  * 발표 및 문서화 : 리드미 결론 수정
#### [류승환](https://github.com/ryuseunghwan1)
  * 데이터 프로세싱 : EDA, SMOTE, ADASYN 샘플러 활용한 CV 진행
  * 발표 및 문서화 : 발표, 리드미 초안 작성
#### [임현수](https://github.com/EbraLim/)
  * 일정 관리 : 일자별/멤버별 업무 기획 및 분배
  * 데이터 프로세싱 : EDA 및 wrong data 발견, borderlineSMOTE 샘플러 활용한 CV 진행
  * 발표 및 문서화 : 프레젠테이션 자료 작성 및 발표, 리드미 최종본 작성

----

project from [Fast campus Datascience School](https://github.com/dss-16th)

(프로젝트 종료 이후 개인 수정되었습니다)
