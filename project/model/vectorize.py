# 라이브러리 호출 파트
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

#################################입력해야 실행 가능####################################
csv_file_path = '<feature_data_path>'
######################################################################################

# 필요항목 추출
feature_df = pd.read_csv(csv_file_path, encoding='cp949')
samples=[]
for index, sample in feature_df['대상'].items():
    samples.append(sample)
    
# BERT 모델과 토크나이저
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 문장 토큰화 및 임베딩 추출
embeddings={}
for sample in tqdm(range(len(samples))):
    sentence = samples[sample]
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    embeddings[sample]=embedding

# 임베딩 값 저장
np.save("embeddings.npy", embeddings)