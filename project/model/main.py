###라이브러리 호출 파트###
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import openai
import requests
from bs4 import BeautifulSoup

###제출 시에는 api키는 삭제###
openai.api_key = "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"

###쿼리인데 이 부분은 사이트랑 연결할 때 자동화해야함###
query_sentence = "희망대출금액 40000000 자산 23000000 수산업자"
job="수산업자"

###파일경로###
csv_file_path_cleaned_data = 'D:\springstudy\kb_ai_challenge\project\model\data.csv'
csv_file_path_info_data = 'D:\springstudy\kb_ai_challenge\project\model\loaninfo(1).csv'


df = pd.read_csv(csv_file_path_cleaned_data, encoding='cp949')

description = pd.read_csv(csv_file_path_info_data, encoding='cp949')

samples=[]
for index, sample in df['대상'].items():
    samples.append(sample)

# BERT 모델과 토크나이저 로드
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# "embeddings.npy" 파일로부터 임베딩값 불러오기
loaded_embeddings = np.load("D:\springstudy\kb_ai_challenge\project\model\embeddings.npy", allow_pickle=True).item()

# 쿼리 문장의 임베딩 추출
def query_vectorize(query_sentence):
    query_inputs = tokenizer(query_sentence, return_tensors="pt")
    with torch.no_grad():
        query_outputs = model(**query_inputs)
    query_embedding = query_outputs.last_hidden_state[:, 0, :].numpy()
    return query_embedding

# 문장 유사도 계산 및 상품별 유사도 합산
def similar_count(query_sentence):
    query_embedding = query_vectorize(query_sentence)
    tray = {}
    for idx, embedding in loaded_embeddings.items():
        similarity = cosine_similarity(query_embedding, embedding)
        index = df.iloc[idx]['상품']
        if index in tray:
            tray[index] += similarity
        else:
            tray[index] = similarity
    sorted_tray = sorted(tray.items(), key=lambda x: x[1], reverse=True)
    return sorted_tray

def top_three(query_sentence):
    result=similar_count(query_sentence)
    rec=[]
    for i in range(3):
        rec.append(result[i][0])
    return rec

def isnan(number,column):
    is_nan = pd.isna(description.loc[number, column])
    return is_nan

def prompt(query_sentence):
    rec = top_three(query_sentence)
    contexts={}
    titles=[]
    for index in rec:
        title,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p = "","","","","","","","","","","","","","","","",""
        if isnan(index, '대출상품명') ==False:
            title=description.loc[index]['대출상품명']+"\n"
        if isnan(index, '대출상품내용') ==False:
            a='대출상품내용: '+description.loc[index]['대출상품내용']+"\n"
        if isnan(index, '최대금액') ==False:
            b='최대금액: '+str(description.loc[index]['최대금액'])+"\n"
        if isnan(index, '대상') ==False:
            c='대상: '+description.loc[index]['대상']+"\n"
        if isnan(index, '상품설명') ==False:
            d= '상품설명: '+description.loc[index]['상품설명']+"\n"
        if isnan(index, '대상상세') ==False:
            e= '대상상세: '+str(description.loc[index]['대상상세'])+"\n"
        if isnan(index, '금리') ==False:
            f= '금리: '+str(description.loc[index]['금리'])+"\n"
        if isnan(index, '대출한도') ==False:
            g= '대출한도: '+str(description.loc[index]['대출한도'])+"\n"
        if isnan(index, '상환방법') ==False:
            h= '상환방법: '+str(description.loc[index]['상환방법'])+"\n"
        if isnan(index, '상환방법상세') ==False:
            i= '상환방법상세: '+str(description.loc[index]['상환방법상세'])+"\n"
        if isnan(index, '금리상세') ==False:
            j= '금리상세: '+str(description.loc[index]['금리상세'])+"\n"
        if isnan(index, '수수료') ==False:
            k= '수수료: '+str(description.loc[index]['수수료'])+"\n"
        if isnan(index, '중도상환수수료') ==False:
            l= '중도상환수수료: '+str(description.loc[index]['중도상환수수료'])+"\n"
        if isnan(index, '필요서류') ==False:
            m= '필요서류: '+str(description.loc[index]['필요서류'])+"\n"
        if isnan(index, '비고') ==False:
            n= '비고: '+str(description.loc[index]['비고'])+"\n"
        if isnan(index, '비고 2') ==False:
            o= '비고 2: '+str(description.loc[index]['비고 2'])+"\n"
        if isnan(index, '링크') ==False:
            p= '링크: '+str(description.loc[index]['링크'])+"\n"
        contexts[title]=a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p
        titles.append(title)
    return contexts, titles

def generate_report(query_sentence):
    contexts, titles = prompt(query_sentence)
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

###기사추천###
def naver_search(query):
    url = f'https://search.naver.com/search.naver?where=news&sm=tab_jum&query={query}'

    # 네이버 검색결과 페이지에 접속
    response = requests.get(url)

    # 접속이 성공적이면 계속 진행
    if response.status_code == 200:
        # BeautifulSoup을 사용하여 HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')
        titles = soup.find_all('a', class_='news_tit')
        
        # Extract the 'href' attribute for each link
        links = [title['href'] for title in titles]
        
        # Extract the text from each title and store it in a list
        title_texts = [title.text for title in titles]
        
        return title_texts, links
    else:
        print(f"검색 결과를 가져오는 데 실패했습니다. 상태 코드: {response.status_code}")
        return []

def generate_newslist(job):
    query=job+" 지원 상품"
    title_texts, links = naver_search(query)
    news_title=title_texts[0:5]
    news_link=links[0:5]
    return news_title, news_link

###리포트 뽑는 방법###
###results,titles=generate_report(query_sentence)
###1순위 추천 상품 제목: titles[0], 2순위 추천 상품 제목: titles[1], 3순위 추천 상품 제목: titles[2]###
###1순위 추천 상품 리포트: results[0], 2순위 추천 상품 리포트: results[1], 3순위 추천 상품 리포트: results[2]###

###기사 뽑는 방법###
###news_title, news_link = generate_newslist(job)
###기사제목: new_title[i], 기사링크: news_link[i]###
###    for i in range(5):
###        print(news_title[i])
###        print(news_link[i])

###sample###
###간단한 결과 확인용 함수입니다! 제출땐 삭제해서 제출예정###
def sample_result(query_sentence,job):
    ###리포트 뽑는 부분###
    results,titles = generate_report(query_sentence)
    print(f"1순위 추천 상품: {titles[0]}")
    print(f"리포트: {results[0]}")
    print(f"2순위 추천 상품: {titles[1]}")
    print(f"리포트: {results[1]}")
    print(f"3순위 추천 상품: {titles[2]}")
    print(f"리포트: {results[2]}")
    ###기사 뽑는 부분###
    news_title, news_link = generate_newslist(job)
    for i in range(len(news_title)):
        print(news_title[i])
        print(news_link[i])
        
sample_result(query_sentence,job)