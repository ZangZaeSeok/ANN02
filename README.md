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

-------------------------------------------------------------------
Temperature: 0.5
and his find something in him

menenius
we are all undone unless
the noble man have mercy

cominius
w
-------------------------------------------------------------------
-------------------------------------------------------------------
Temperature: 0.6
a goodly son
to be your comforter when he is gone

queen elizabeth
as much to you good sister whither
-------------------------------------------------------------------
-------------------------------------------------------------------
Temperature: 0.7
ad a taste of his obedience
our aediles smote ourselves resisted come

menenius
consider this he has 
-------------------------------------------------------------------
-------------------------------------------------------------------
Temperature: 0.7999999999999999
astings help me to my closet
oh poor clarence

gloucester
this is the fruit of rashness markd you not
-------------------------------------------------------------------
-------------------------------------------------------------------
Temperature: 0.8999999999999999
ake with unthankfulness his doing
in common worldly things tis calld ungrateful
with dull unwillignes
-------------------------------------------------------------------
-------------------------------------------------------------------
Temperature: 0.9999999999999999
an misery itself would give rewards
his deeds with doing them and is content
to spend the time to end
-------------------------------------------------------------------
-------------------------------------------------------------------
Temperature: 1.0999999999999999
ave you lunkd
that thou wilt war with god by murdering me
ah sirs consider he that set you on
to do t
-------------------------------------------------------------------
-------------------------------------------------------------------
Temperature: 1.1999999999999997
ak good cominius
leave notice that no manner of person
at any time have recourse upon the ice
or hail
-------------------------------------------------------------------
-------------------------------------------------------------------
Temperature: 1.2999999999999998
a place for you pray you avoid come

coriolanus
follow your function go and batten on cold bits

thir
-------------------------------------------------------------------
-------------------------------------------------------------------
Temperature: 1.4
ave
not been common in my love i will sir flatter my
sworn brother the people to earn a dearer
estima
-------------------------------------------------------------------
-------------------------------------------------------------------
Temperature: Down
ay

tyrrelse
nor ever will be ruled

brutus
callt not a plot
the people cry you mockd them and of lat
-------------------------------------------------------------------
-------------------------------------------------------------------
Temperature: Up
as thou beat me
and wouldst do so i think should we encounter
as often as we eat by the elements
if e
-------------------------------------------------------------------
