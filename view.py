import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import feature
import path
import os

class ImageSearch:
    fe = feature.FeatureExtractor()
    def __init__(self):
        self.features, self.img_paths = self.load_database()
        self.fe = feature.FeatureExtractor()

    def load_database(self):
        # 데이터베이스 이미지의 특성과 경로를 로드
        df = pd.read_csv('pokemon.csv')
        names = df['Name'].tolist()
        features = []
        img_paths = []

        for n in names:
            if os.path.exists(f"./images/images/{n}.png"):
                image_path = f"./images/images/{n}.png"
            else:
                image_path = f"./images/images/{n}.jpg"

            img_paths.append(image_path)
            feature = self.fe.extract(img=Image.open(image_path))
            features.append(feature)

        return np.array(features), img_paths

    def search_similar_images(self, query_img_path):
        # Insert the image query
        img = Image.open(query_img_path)
        # 이미지 특성 추출
        query = self.fe.extract(img)

        # 데이터베이스 이미지와 검색 이미지 사이의 거리를 계산
        dists = np.linalg.norm(self.features - query, axis=1)

        # 가장 유사한 30개의 이미지 인덱스를 가져옴
        ids = np.argsort(dists)[:30]
        # 유사도, 이미지 경로 및 인덱스를 포함하는 튜플을 생성
        scores = [(dists[id], self.img_paths[id], id) for id in ids]

        #Visualize the result
        axes=[]
        fig=plt.figure(figsize=(8,8))
        for a in range(5*6):
            score = scores[a]
            axes.append(fig.add_subplot(5, 6, a+1))
            subplot_title=str(round(score[0],2)) + "/m" + str(score[2]+1)
            axes[-1].set_title(subplot_title)  
            plt.axis('off')
            plt.imshow(Image.open(score[1]))
        fig.tight_layout()
        plt.show()

# 클래스 인스턴스 생성
search_engine = ImageSearch()

# 검색할 이미지 경로 설정
query_img_path = "./test/ma.jpg"

# 유사한 이미지 검색
search_engine.search_similar_images(query_img_path)
