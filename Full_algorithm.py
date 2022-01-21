# ����� �������� ��������� ��������� ���������� ���� �����
#!pip install gensim 

# ����������� ������ ����������
from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# �������� �������. ������ �������� - ���������� �����, ������ - ������ ����� (���-�� ������� � ����� �����)
def split(file_path, batch_size):
    
    # ������� ��� ����������� �������� �������� ��� �� ���� ����������
    def encode_weight(x):
        x = float(x)
        if x < 300:
            return '0_300'
        if x < 500:
            return '300_500'
        return '500_1000'
    
    # ������� ��� ����������� �������� �������� ��������� �� ���� ����������
    def encode_cost(x):
        x = float(x)
        if x < 2.6:
            return '0_3'
        if x < 5:
            return '3_5'
        return '5_10'
    
    # ������� ������� ����� ��������������� ��������
    def get_vector(e):
        for elem in variants:
            if elem['diamond'] == e:
                return elem['vector']
            
    # ������ ������ ��� ������ word2vec
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except:
        df = pd.read_csv(file_path)
    
    df = df.drop(['�����', ], axis=1)
    
    encoders = {'�����': ['������', '����', '�������', '�����������'], '����': ['C����', '�������', '������', 'Ƹ����'], '������': ['�������', '�������', '�����'], '�������������': ['��������', '�� ��������'], '���': ['300_500', '500_1000', '0_300'], '���������': ['3_5', '5_10', '0_3']}
    
    # �������� ������� �����������
    df['���'] = df['���'].apply(encode_weight)
    df['���������'] = df['���������'].apply(encode_cost)
    
    # �������� numpy array �������, ����� ������ ������� ���������
    data = [list(x) for x in np.array(df)]
    
    # ������ ������
    model = Word2Vec(vector_size=16, min_count=1)
    model.build_vocab(data)
    model.train(data, total_examples=model.corpus_count, epochs=100, report_delay=1)

    # � ������ variants ��� ������ diamond ����� ��������� ��������� ������, � ��� vector ��� ������
    variants = []
    for a in encoders['�����']:
        for b in encoders['����']:
            for c in encoders['������']:
                for d in encoders['�������������']:
                    for e in encoders['���']:
                        for f in encoders['���������']:
                            variants.append({'diamond': [a, b, c, d, e, f], 'vector': list((model.wv[a] +model.wv[b] + model.wv[c] + model.wv[d] + model.wv[e] + model.wv[f]).astype(float))})

    # ������ �����
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except:
        df = pd.read_csv(file_path) 
    
    # ������� ����� �������� �������
    embeddings = deepcopy(df)
    embeddings['���'] = embeddings['���'].apply(encode_weight)
    embeddings['���������'] = embeddings['���������'].apply(encode_cost)
    
    # ������� ������� ��� ������� ������ ������ ������������� ��� ����������������� ��������
    id_vector = {i: get_vector(list(embeddings[embeddings['�����'] == i].iloc[0][1:])) for i in embeddings['�����']}
    
    batchs = []
    
    # ��������� �� ���������� �� ������� �������. ������ (len(df) // batch_size) ��������� � ������ batchs. 
    # � ����� � batchs ����� �������� ������ ������� ������� �������
    for i in range(batch_size):
        diamond = list(embeddings.iloc[0])
        distance = cosine_similarity([id_vector[diamond[0]]],
                                     [id_vector[id_] for id_ in embeddings['�����']])
        embeddings['distance'] = distance[0].reshape(-1, 1)
        embeddings.sort_values(by=['distance'], inplace=True)
        embeddings = embeddings.iloc[::-1]
        embeddings.reset_index(inplace=True, drop=True)
        batchs.append([])
        cache = []
        for j in range(len(df) // batch_size):
            batchs[i].append(embeddings.iloc[j]['�����'])
            cache.append(j)
        embeddings.drop(cache, axis=0, inplace=True)
        embeddings.reset_index(inplace=True, drop=True)
    
    # ����� � ������ ����� ���� ������������� ������ (� � ������ batchs �������� ������ ������� ������� �������)
    # ���������� �� ����� batchs � � ������ batch �������� i-�� ������� ���� ���������� batchs
    # ��������� ����� ������ � ��������� ����
    for i in range(len(df) // batch_size):
        batch = [batchs[j][i] for j in range(batch_size)]
        df[df['�����'].isin(batch)].to_excel(f'box_{i+1}.xls', index=False)
        
    # � ����� �������� ������ ��� ������� ����� ��������� excel ����
    
# ��� ������� �������
# split(���� � �����, ���������� ������� � ����� �����)