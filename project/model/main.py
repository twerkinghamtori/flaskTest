# 라이브러리 호출 파트
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import openai
import requests
from bs4 import BeautifulSoup

#################################입력해야 실행 가능####################################
openai.api_key = "<API KEY>"

# 파일경로
csv_file_path_feature_data = "<feature_data_path>"
csv_file_path_info_data = "<loaninfo_data_path>"
######################################################################################

# 데이터 호출
feature_df = pd.read_csv(csv_file_path_feature_data, encoding='cp949') 
info_df = pd.read_csv(csv_file_path_info_data, encoding='cp949')

# BERT 모델과 토크나이저
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# "embeddings.npy" 파일로부터 임베딩값 호출
loaded_embeddings = np.load("embeddings.npy", allow_pickle=True).item()

# 쿼리 문장의 임베딩값 추출
def query_vectorize(query_sentence):
    query_inputs = tokenizer(query_sentence, return_tensors="pt")
    with torch.no_grad():
        query_outputs = model(**query_inputs)
    query_embedding = query_outputs.last_hidden_state[:, 0, :].numpy()
    return query_embedding

# 문장 유사도 계산 및 상품별 유사도 합산
def similar_count(query_sentence1,query_sentence2,query_sentence3):
    query_embedding1 = query_vectorize(query_sentence1)
    query_embedding2 = query_vectorize(query_sentence2)
    query_embedding3 = query_vectorize(query_sentence3)
    tray = {}
    for idx, embedding in loaded_embeddings.items():
        similarity1 = cosine_similarity(query_embedding1, embedding)
        similarity2 = cosine_similarity(query_embedding2, embedding)
        similarity3 = cosine_similarity(query_embedding3, embedding)
        index = feature_df.iloc[idx]['상품']
        if index in tray:
            tray[index] += similarity1
            tray[index] += similarity2
            tray[index] += similarity3
        else:
            tray[index] = similarity1
    sorted_tray = sorted(tray.items(), key=lambda x: x[1], reverse=True)
    return sorted_tray

# 유사도 점수 1,2,3순위 상품의 Label 추출
def top_three(query_sentence1,query_sentence2,query_sentence3):
    sorted_tray=similar_count(query_sentence1,query_sentence2,query_sentence3)
    rec=[]
    for i in range(3):
        rec.append(sorted_tray[i][0])
    return rec

# df에 결측치가 포함되었을때 True를 반환
def isnan(number,column):
    is_nan = pd.isna(info_df.loc[number, column])
    return is_nan

# Chat GPT에 사용할 프롬포트 생성
def prompt(query_sentence1,query_sentence2,query_sentence3):
    rec = top_three(query_sentence1,query_sentence2,query_sentence3)
    contexts={}
    titles=[]
    for index in rec:
        title,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p = "","","","","","","","","","","","","","","","",""
        if isnan(index, '대출상품명') ==False:
            title=info_df.loc[index]['대출상품명']+"\n"
        if isnan(index, '대출상품내용') ==False:
            a='대출상품내용: '+info_df.loc[index]['대출상품내용']+"\n"
        if isnan(index, '최대금액') ==False:
            b='최대금액: '+str(info_df.loc[index]['최대금액'])+"\n"
        if isnan(index, '대상') ==False:
            c='대상: '+info_df.loc[index]['대상']+"\n"
        if isnan(index, '상품설명') ==False:
            d= '상품설명: '+info_df.loc[index]['상품설명']+"\n"
        if isnan(index, '대상상세') ==False:
            e= '대상상세: '+str(info_df.loc[index]['대상상세'])+"\n"
        if isnan(index, '금리') ==False:
            f= '금리: '+str(info_df.loc[index]['금리'])+"\n"
        if isnan(index, '대출한도') ==False:
            g= '대출한도: '+str(info_df.loc[index]['대출한도'])+"\n"
        if isnan(index, '상환방법') ==False:
            h= '상환방법: '+str(info_df.loc[index]['상환방법'])+"\n"
        if isnan(index, '상환방법상세') ==False:
            i= '상환방법상세: '+str(info_df.loc[index]['상환방법상세'])+"\n"
        if isnan(index, '금리상세') ==False:
            j= '금리상세: '+str(info_df.loc[index]['금리상세'])+"\n"
        if isnan(index, '수수료') ==False:
            k= '수수료: '+str(info_df.loc[index]['수수료'])+"\n"
        if isnan(index, '중도상환수수료') ==False:
            l= '중도상환수수료: '+str(info_df.loc[index]['중도상환수수료'])+"\n"
        if isnan(index, '필요서류') ==False:
            m= '필요서류: '+str(info_df.loc[index]['필요서류'])+"\n"
        if isnan(index, '비고') ==False:
            n= '비고: '+str(info_df.loc[index]['비고'])+"\n"
        if isnan(index, '비고 2') ==False:
            o= '비고 2: '+str(info_df.loc[index]['비고 2'])+"\n"
        if isnan(index, '링크') ==False:
            p= '링크: '+str(info_df.loc[index]['링크'])+"\n"
        contexts[title]=a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p
        titles.append(title)
    return contexts, titles

# Chat GPT API를 활용한 리포트 생성
def generate_report(query_sentence1,query_sentence2,query_sentence3):
    contexts, titles = prompt(query_sentence1,query_sentence2,query_sentence3)
    results=[]
    for context in contexts:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "주어지는 대출 상품에 대한 설명을 바탕으로, 금융상품 분석 리포트를 500자 이상으로, 존댓말로 작성해줘"},
                    {"role": "user", "content": contexts[context]}
                    ]
                )
            result = completion.choices[0].message
            result=result["content"]
            results.append(result)
        except:
            pass
    return results,titles

# 관련 기사 목록 크롤링
def naver_search(query):
    url = f'https://search.naver.com/search.naver?where=news&sm=tab_jum&query={query}'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        titles = soup.find_all('a', class_='news_tit')
        links = [title['href'] for title in titles]
        title_texts = [title.text for title in titles]
        return title_texts, links
    else:
        return []

# 관련 기사 목록 추출
def generate_newslist(job):
    query=job+" 지원 상품"
    title_texts, links = naver_search(query)
    news_title=title_texts[0:5]
    news_link=links[0:5]
    return news_title, news_link

# 리포트 뽑는 방법
# results,titles=generate_report(query_sentence)
# 1순위 추천 상품 제목: titles[0], 2순위 추천 상품 제목: titles[1], 3순위 추천 상품 제목: titles[2]
# 1순위 추천 상품 리포트: results[0], 2순위 추천 상품 리포트: results[1], 3순위 추천 상품 리포트: results[2]

# 기사 뽑는 방법
# news_title, news_link = generate_newslist(job)
# 기사제목: new_title[i], 기사링크: news_link[i]
