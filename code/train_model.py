# 2_train_model.py

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

print("Đang tải dữ liệu embeddings...")
# Tải dữ liệu đã được xử lý
data = np.load('faces_embeddings.npz')
X_train, y_train = data['arr_0'], data['arr_1']
print("Đã tải xong!")

# Mã hóa nhãn (chuyển tên người thành số)
print("Đang mã hóa nhãn...")
encoder = LabelEncoder()
encoder.fit(y_train)
y_train_encoded = encoder.transform(y_train)

# Huấn luyện mô hình SVM
print("Đang huấn luyện mô hình SVM...")
# kernel='linear' và C=1.0 là các tham số phổ biến, probability=True để lấy xác suất dự đoán
model = SVC(kernel='linear', probability=True, C=1.0)
print(f"Unique classes in y_train_encoded: {np.unique(y_train_encoded)}")
print(f"Number of unique classes: {len(np.unique(y_train_encoded))}")
model.fit(X_train, y_train_encoded)
print("Hoàn tất huấn luyện!")

# Lưu mô hình SVM đã huấn luyện
model_filename = 'models/svm_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print(f"Đã lưu mô hình SVM vào file: {model_filename}")

# Lưu bộ mã hóa nhãn
labels_filename = 'models/labels.pkl'
with open(labels_filename, 'wb') as f:
    pickle.dump(encoder, f)
print(f"Đã lưu bộ mã hóa nhãn vào file: {labels_filename}")