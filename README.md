# analysis_korean_political_news

이화여대미래혁신센터 13기 도전학기

---

## 주제 : 자연어처리 기술을 활용한 정치 뉴스 기사의 비편향적 접근법 제시

설명 : 20대 대선과 관련한 정치 뉴스 기사를 자연어처리 기술을 활용하여 키워드 별 트렌드를 분석하여 새로운 분류 기준을 도입, 대중에게 비편향적 접근법을 제시하기 위한 연구를 진행하였음.

1. data

     - 수집한 raw 데이터

     - 모델링 결과를 csv 로 저장

2. crawling

     - crawling을 이용하여 뉴스 중 정치(선거) 카테고리 데이터 확보

     - Github action 으로 crawler 제장

3. topic_modeling
     - LDA 알고리즘 적용하여 기사 내 Topic을 추출

4. topic_classification

     - 경제/청년/여성/환경 4개의 핵심어 별 기사 분류

     - Klue YNAT task 활용

     - model architecture

5. named_entity_recognition

     - 개체명 인식 기술을 적용하여(Named entity recognition) 기사의 내용을 “인물/장소/기관/키워드”로 분류

     - Klue NER task 활용

     - model architecture

6. relation_extraction

     - 관계 추출 기술을 적용하여(Relation Extraction) 기사의 내용을 “주어/목적어/서술어”로 분류

     - Klue RE task 활용

     - model architecture

7. sentiment_analysis

     - 기사의 내용을 “positive/neutral/negative”로 분류

     - model architecture

8. model serving

     - Streamlit을 활용하여 model serving

     - 동적 시각화 웹 사이트 배포
