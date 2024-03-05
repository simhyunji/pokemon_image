import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class FeatureExtractor:
    def __init__(self):
        # VGG-16 아키텍처 및 ImageNet 가중치를 사용
        base_model = VGG16(weights='imagenet')
        # 모델을 수정하여 완전 연결 레이어에서 특징을 반환
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        # 이미지 크기 조정
        img = img.resize((224, 224))
        # 이미지 색 공간 변환
        img = img.convert('RGB')
        # 이미지 형식 재구성
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # 특징 추출
        feature = self.model.predict(x)[0]
        # 특징 벡터를 단위 벡터로 정규화하여 반환
        return feature / np.linalg.norm(feature)
