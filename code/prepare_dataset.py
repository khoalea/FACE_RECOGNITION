# 1_prepare_dataset.py

import os
from os import listdir
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet

# import matplotlib.pyplot as plt
import cv2 # Giả sử ảnh gốc được đọc bằng OpenCV

# Khởi tạo các model
detector = MTCNN()
embedder = FaceNet()

# Hàm để trích xuất một khuôn mặt từ ảnh
def extract_face(filename, required_size=(160, 160)):
    # Đọc ảnh từ file
    image = Image.open(filename)
    # Chuyển sang dạng RGB, nếu cần
    image = image.convert('RGB')
    # Chuyển sang dạng mảng numpy
    pixels = np.asarray(image)
    # Dùng MTCNN để phát hiện khuôn mặt
    results = detector.detect_faces(pixels)
    
    # Kiểm tra xem có khuôn mặt nào được phát hiện không
    if len(results) == 0:
        return None
        
    # Lấy bounding box từ khuôn mặt đầu tiên
    x1, y1, width, height = results[0]['box']
    # Xử lý tọa độ âm
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    # Cắt lấy khuôn mặt
    face = pixels[y1:y2, x1:x2]
    cv2.imshow('Extracted Face', face)
    cv2.waitKey(0)
    # Resize khuôn mặt về kích thước yêu cầu (160x160)
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    
    return face_array

# Hàm để tải dataset và trích xuất embeddings
def load_dataset(directory):
    X, y = list(), list()
    # Duyệt qua các thư mục con (mỗi thư mục là một người)
    for subdir in listdir(directory):
        path = os.path.join(directory, subdir)
        # Bỏ qua nếu không phải là thư mục
        if not os.path.isdir(path):
            continue
        
        # Tải tất cả các khuôn mặt trong thư mục con
        faces = list()
        for filename in listdir(path):
            filepath = os.path.join(path, filename)
            face = extract_face(filepath)
            if face is not None:
                faces.append(face)
        
        # Nếu có khuôn mặt được tải, tạo embeddings
        if len(faces) > 0:
            # Dùng FaceNet để lấy embeddings
            embeddings = embedder.embeddings(faces)
            # Lưu embeddings và nhãn
            for embedding in embeddings:
                X.append(embedding)
                y.append(subdir) # Tên thư mục chính là nhãn
            print(f"> Đã tải {len(faces)} ảnh cho lớp: {subdir}")

    return np.asarray(X), np.asarray(y)

# Đường dẫn đến thư mục dataset
dataset_path = 'dataset/'

# Tải dataset và tạo embeddings
print("Đang tải dataset và tạo embeddings...")
X_train, y_train = load_dataset(dataset_path)
print("Hoàn tất!")
print("Kích thước dữ liệu X:", X_train.shape)
print("Kích thước dữ liệu y:", y_train.shape)

# Lưu embeddings và nhãn vào một file nén
np.savez_compressed('faces_embeddings.npz', X_train, y_train)
print("Đã lưu embeddings vào file 'faces_embeddings.npz'")