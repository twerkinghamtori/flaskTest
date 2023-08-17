from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import openai  # openai 모듈을 import
import sys
from flask import Flask, render_template, request
from project.model import main

application = Flask(__name__)

openai.api_key = "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"

@application.route("/")
def main_page():
    return render_template("main_page.html")

@application.route("/info")
def info():
    return render_template("info.html")

@application.route("/report", methods=['POST', 'GET'])
def report(): 
    if request.method == 'POST':
        data = request.form
        name = data.get('name') #이름
        age = data.get('age') #나이
        family = data.getlist('family') #가족구성원
        family_members = ', '.join(family)
        job = data.get('job') #종사직
        loan = data.get('loan') #기존대출여부
        loan_status = "없" if loan == "n" else "있"
        
        query_sentence = f"나이는 {age}인 {job}인이고 가족 구성원은 {family_members}이며, 기존대출내역은 {loan_status}고,  "
        
        if loan == "y":
            loan_name = data.get('loan_name') #상품명
            loan_time = data.get('loan_time') #상환 잔여 기간
            loan_rate = data.get('loan_rate') #현재 이자율
            
            query_sentence += f" 상품명은 {loan_name}, 상환 잔여 기간은 {loan_time}개월, 현재 이자율은 {loan_rate}% 입니다."
         
        asset = data.get('asset') #자산규모
        certificate = data.get('certificate') #공인인증서 사용여부
        certificate_status = "사용하지않고" if certificate == "n" else "사용하고"
        others = data.get('others') #기타 특이사항
        
        query_sentence += f"자산규모는 {asset}원이고, 공인인증서를 {certificate_status}, 기타 특이사항은 {others}인 {job}인" 
        
        job_query = f"{job}인"       
        
        #출력부분(나중에 지워야함)
        original_stdout = sys.stdout  # 기존 stdout 저장
        sys.stdout = sys.__stdout__  # stdout을 원래의 값으로 설정

        print("query_sentence:", query_sentence)  # 터미널에 출력

        sys.stdout = original_stdout  # stdout을 원래 값으로 복원
        print("job_query:", job_query)
        
        # AI 모델 함수 실행
        results, titles = main.generate_report(query_sentence)

        # 기사 추천 결과 생성
        news_title, news_link = main.generate_newslist(job_query)
        
        return render_template("report.html", name=name, title=titles, result=results, news_titles=news_title, news_links=news_link)
    return render_template('report.html', name=name, title=titles, result=results, news_titles=news_title, news_links=news_link)

if __name__ == "__main__" : 
    application.run(host="0.0.0.0", port=9900)