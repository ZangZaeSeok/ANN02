# ANN02
인공 신경망 과제2

## report
- report.ipynb: 과제와 연관하여 코드가 수행된 jupyter notebook

## template
### dataset.py
- dataset 코드가 작성된 파일
- 특수 문자로 인한 잘못된 학습을 방지하기 위해 특수 문자 제거
- 알파벳 소/대문자 혼용으로 인한 잘못된 학습을 방지하기 위해 소문자화 수행
  
### model.py
- 모델 코드
- RNN & LSTM code 확인 가능

### main.py
- 모델 학습 및 validation 수행
- validation 성능이 가장 좋은 시점을 저장

### generate.py
- 모델, seed character, temperature을 입력 받으면 길이가 100인 문장 생성
- *generate_down: temperature의 분산을 낮추면서 생성
- *generate_up: temperature의 분산을 높이면서 생성

# 결과
## Train & Valid Graph
- report.ipynb에서 확인 가능
  

## 문장 생성 결과
- report.ipynb에서 확인 가능
- LSTM 모델의 가장 좋은 시점을 불러와서 생성 Test
### 'a'가 seed_char일 때 chat gpt를 통한 가장 좋은 temperature 검사 결과

'''
다음 중 가장 말이 되는 부분은 온도 0.5입니다:

```
Temperature: 0.5
and his find something in him

menenius
we are all undone unless
the noble man have mercy

cominius
w
```

이 부분은 Shakespeare의 대화 스타일을 가장 잘 유지하면서도 문장의 흐름이 어느 정도 자연스럽습니다. 온도가 낮을수록 언어 모델의 출력이 원본 텍스트와 더 유사해지며, 문장의 일관성이 높아지는 경향이 있습니다.
'''
