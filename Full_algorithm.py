# перед запуском алгоритма загрузите пожалуйста этот пакет
#!pip install gensim 

# импортируем нужные библиотеки
from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# основная функция. первый аргумент - директория файла, второй - размер бокса (кол-во алмазов в одном боксе)
def split(file_path, batch_size):
    
    # функция для кодирования значений признака Вес по трем категориям
    def encode_weight(x):
        x = float(x)
        if x < 300:
            return '0_300'
        if x < 500:
            return '300_500'
        return '500_1000'
    
    # функция для кодирования значений признака Стоимость по трем категориям
    def encode_cost(x):
        x = float(x)
        if x < 2.6:
            return '0_3'
        if x < 5:
            return '3_5'
        return '5_10'
    
    # функция которая берет векторизованные признаки
    def get_vector(e):
        for elem in variants:
            if elem['diamond'] == e:
                return elem['vector']
            
    # читаем данные для модели word2vec
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except:
        df = pd.read_csv(file_path)
    
    df = df.drop(['Номер', ], axis=1)
    
    encoders = {'Форма': ['Звезда', 'Круг', 'Квадрат', 'Треугольник'], 'Цвет': ['Cиний', 'Красный', 'Зелёный', 'Жёлтый'], 'Размер': ['Большой', 'Средний', 'Малый'], 'Флуоресценция': ['Светится', 'Не светится'], 'Вес': ['300_500', '500_1000', '0_300'], 'Стоимость': ['3_5', '5_10', '0_3']}
    
    # применим функции кодирования
    df['Вес'] = df['Вес'].apply(encode_weight)
    df['Стоимость'] = df['Стоимость'].apply(encode_cost)
    
    # создадим numpy array матрицу, чтобы модель быстрее обучилась
    data = [list(x) for x in np.array(df)]
    
    # обучим модель
    model = Word2Vec(vector_size=16, min_count=1)
    model.build_vocab(data)
    model.train(data, total_examples=model.corpus_count, epochs=100, report_delay=1)

    # в списке variants под ключом diamond будут храниться эмбединги алмаза, а под vector его вектор
    variants = []
    for a in encoders['Форма']:
        for b in encoders['Цвет']:
            for c in encoders['Размер']:
                for d in encoders['Флуоресценция']:
                    for e in encoders['Вес']:
                        for f in encoders['Стоимость']:
                            variants.append({'diamond': [a, b, c, d, e, f], 'vector': list((model.wv[a] +model.wv[b] + model.wv[c] + model.wv[d] + model.wv[e] + model.wv[f]).astype(float))})

    # читаем файлы
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except:
        df = pd.read_csv(file_path) 
    
    # создаем копию основной таблицы
    embeddings = deepcopy(df)
    embeddings['Вес'] = embeddings['Вес'].apply(encode_weight)
    embeddings['Стоимость'] = embeddings['Стоимость'].apply(encode_cost)
    
    # создаем словарь где каждому номеру алмаза соответствует его векторизированные признаки
    id_vector = {i: get_vector(list(embeddings[embeddings['Номер'] == i].iloc[0][1:])) for i in embeddings['Номер']}
    
    batchs = []
    
    # сортируем по расстоянию от первого обьекта. первые (len(df) // batch_size) добавляем в список batchs. 
    # в итоге в batchs будут хранится списки номеров похожих алмазов
    for i in range(batch_size):
        diamond = list(embeddings.iloc[0])
        distance = cosine_similarity([id_vector[diamond[0]]],
                                     [id_vector[id_] for id_ in embeddings['Номер']])
        embeddings['distance'] = distance[0].reshape(-1, 1)
        embeddings.sort_values(by=['distance'], inplace=True)
        embeddings = embeddings.iloc[::-1]
        embeddings.reset_index(inplace=True, drop=True)
        batchs.append([])
        cache = []
        for j in range(len(df) // batch_size):
            batchs[i].append(embeddings.iloc[j]['Номер'])
            cache.append(j)
        embeddings.drop(cache, axis=0, inplace=True)
        embeddings.reset_index(inplace=True, drop=True)
    
    # чтобы в каждом боксе были разнообразные алмазы (а в списке batchs хранятся списки номеров ПОХОЖИХ алмазов)
    # проходимся по циклу batchs и в списке batch передаем i-ые индексы всех подсписков batchs
    # сохраняем боксы каждый в отдельный файл
    for i in range(len(df) // batch_size):
        batch = [batchs[j][i] for j in range(batch_size)]
        df[df['Номер'].isin(batch)].to_excel(f'box_{i+1}.xls', index=False)
        
    # в итоге алгоритм создал для каждого бокса отдельный excel файл
    
# как вызвать функцию
# split(путь к файлу, количество алмазов в одном боксе)