# 판례 추천 프로젝트(민법)
## 개요
* 민법 조항과 민법에 관련된 판례에 대한 정보를 제공하는 서비스
* 검색 기능, 카테고리 기능, 판례 추천 기능으로 구성

## 데이터
* 학습 데이터는 아래에 따로 기재
* 민법 데이터: 국회법률정보시스템에서 다운로드 받아 사용
* 판례 데이터: 법제처의 국가법령 공동활용센터의 판례 목록 api와 판례 본문 api 이용. 
  * 판례 목록 api에 키퉈드를 넣어 판례 일련번호 리스트를 불러오고, 일련번호를 판례 본문 api에 넣어 본문 저장.

## 검색 기능과 카테고리 기능
* 검색 기능: 입력된 키워드가 포힘된 민법 조항과 판례를 보여준다.
* 카테고리 기능: 민법과 판례를 카테고리화하여 보여준다.
  * 상위 카테고리와 하위 카테고리로 나누었으며 상위 카테고리는 민법의 각 편으로, 하위 카테고리는 해당 편의 각 장으로 하였다.
    * 민법의 제 1편은 총칙이고 제 1편의 제 1장은 통칙, 제 2장은 인, 제 3장은 법인, 제 4장은 물건, 제 5장은 법률행위, 제 6장은 기간, 제 7장은 소멸시효이다. 따라서 '총칙'은 첫 번째 상위 카테고리로, '통칙, 인, 법인, 물건, 법률행위, 기간, 소멸시효'는 총칙의 하위 카테고리로 선정하였다.
* 검색 화면
  ![Image](https://github.com/user-attachments/assets/d372a998-ed96-463b-956d-c93fb487db2d)
* 카테고리별 민법 화면
  ![Image](https://github.com/user-attachments/assets/12a66739-e0a1-4035-9a87-5fc123666a84)
## 판례 추천 기능
* 기능 소개
  * 사용자가 자신이 처한 상황을 입력하면 가장 유사하다고 판단되는 판례 다섯개를 추천해주는 기능
  * kobert를 이용하여 분류를 먼저 수행한 후 해당 판례들 중에 유사도 검사를 수행하여 판례를 추천한다. 이때 판례 중에서 판결요지 부분을 kobart로 요약하여 보여주고, 이를 클릭하면 전체 판례 내용을 보여준다.
* 학습 데이터
  * kobert 학습할 때 법률 구조 공단의 상담 데이터를 사용하였고, 판례 데이터도 이와 맞추기 위해 법률 구조 공단의 카테고리를 이용하여 판례를 불러왔다.
    * 법률 구조 공단의 제목과 질문을 이용하였는데, 제목은 ‘미성년자 단독으로 책을 할부구입한 때 이를 취소할 수 있는지’와 같이 질문이 요약된 형태이고 질문은 상황을 자세히 설명하는 형태였다. 
  * kobart 학습할 때 ai 허브의 요약문 및 레포트 생성 데이터를 사용하였다.
* kobert 파인튜닝 파라미터 선택
  * 베이스라인: max_len 110(제목과 질문 전체의 평균 길이), epoch 15, batch_size 32, learning_rate 0.00005(5e-5)
  * 15 epoch까지의 결과 중 검증 데이터에 대한 정확도가 가장 높았던 것을 기준으로 비교했다.
  * 베이스라인 정확도: 0.8458
  * max_len: 제목과 길이 전체의 평균 길이를 해보니 110이었다. 110, 200, 300까지 해보고 가장 정확도가 높았던 200으로 결정
  * batch_size: 16, 32, 64를 비교해본 결과 가장 정확도가 높았던 32로 결정
  * learning_rate: 5e-5, 3e-5, 1e-5를 비교해본 결과 가장 정확도가 높았던 1e-5로 결정
  * 최종적으로, 검증 데이터에 대한 정확도가 0.8458에서 0.8569로 약 1.1% 상승했다.
* 판례 추천 화면
  ![Image](https://github.com/user-attachments/assets/34b0f466-0e97-4a30-97b6-a4250056a466)

## 개선해야 될 부분
* 정확도 개선
  * 판례 데이터 따로 질문 데이터 따로 얻다 보니 둘의 관계성이 없어서 추천 결과의 정확도를 확인하기 어려웠고, 결국 정확도를 제대로 높이지 못하였다.
* 답변을 이해하기 어려움
  * 판례 내용을 kobart로 요약해서 출력하긴 하였으나 유사도가 높은 판례 내용을 제시하는 것에 불과해서 질문에 대한 답을 얻으려면 판례 내용을 읽고 이해해야 했다.
* 이를 개선하기 위한 방안을 생각하다, 추천 기능을 발전시켜 RAG 기반의 챗봇 형태로 구현한 법률 질의응답 챗봇(law_qna_chatbot) 프로젝트를 진행하였다.

## 배포
* gunicorn과 nginx를 이용하여 배포를 진행하였다.
* /etc/nginx/sites-available/myapp.conf 를 아래와 같이 설정했다.
  <pre><code>
    server {

        listen 80;
        location = /favicon.ico { access_log off; log_not_found off; }
        access_log /var/log/nginx/access.log;
        location / {
          root /var/www/myapp/build;
          index  index.html index.htm;
          try_files $uri $uri/ /index.html;
        }
        location ^~ /api {
          proxy_pass http://172.18.50.37:5000/api;
          proxy_redirect off;
          proxy_buffer_size          128k;
          proxy_buffers              4 256k;
          proxy_busy_buffers_size    256k;

        }
    }
  </code></pre>
* nginx 실행
  <pre><code>sudo service nginx start</code></pre>
* gunicorn 실행
  <pre><code>gunicorn -w 1 --timeout 120 --bind 0.0.0.0:5000 wsgi:app</code></pre>

## 파일 설명
* data.ipynb： 판례 데이터와 민법 데이터, 학습 데이터 저장
* app.py: 파이썬 플라스크 전체 코드
* kobert.ipynb: kobert 학습 코드
* 요약
  * kobart_train.py: kobart 학습 코드
  * kobart.ipynb: kobart 학습 실행 코드
  * summary.ipynb: 학습된 kobart로 판결 요지 요약하는 코드
* 리액트
  * search.js: 검색 페이지
  * search_title.js. search_conts.js: 검색 페이지에 사용되는 컴포넌트
  * search_deatail.js: 검색 상세 페이지
  * category.js: 카테고리 페이지
  * categort_conts.js: 카테고리 페이지에 사용되는 컴포넌트
  * recommand.js: 판례 추천 페이지
