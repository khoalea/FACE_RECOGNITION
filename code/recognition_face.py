# 3_recognize_face.py

import cv2
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import pickle

# --- TẢI CÁC MODEL CẦN THIẾT ---
print("Đang tải các model...")
# Model MTCNN để phát hiện khuôn mặt
detector = MTCNN()
# Model FaceNet để trích xuất embedding
embedder = FaceNet()
# Model SVM để nhận diện
with open('models/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
# Bộ mã hóa nhãn
with open('models/labels.pkl', 'rb') as f:
    encoder = pickle.load(f)
print("Đã tải xong các model!")

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    # Đọc một khung hình từ webcam
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc khung hình từ webcam.")
        break

    # Dùng MTCNN để phát hiện khuôn mặt
    faces = detector.detect_faces(frame)
    print(faces)
    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        
        # Cắt khuôn mặt từ khung hình
        face_pixels = frame[y1:y2, x1:x2]
        
        # Resize về 160x160
        image = Image.fromarray(face_pixels)
        image = image.resize((160, 160))
        face_array = np.asarray(image)

        # Lấy embedding của khuôn mặt
        face_embedding = embedder.embeddings([face_array])

        # Dự đoán bằng mô hình SVM
        # Model SVM yêu cầu đầu vào 2D, nên ta reshape
        prediction = svm_model.predict(face_embedding)
        probability = svm_model.predict_proba(face_embedding)
        
        # Lấy tên và xác suất
        person_name = encoder.inverse_transform(prediction)[0]
        confidence = np.max(probability) * 100
        
        # Hiển thị kết quả lên khung hình
        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Chuẩn bị text để hiển thị
        if confidence > 65.0: # Đặt một ngưỡng tin cậy
            text = f'{person_name} ({confidence:.2f}%)'
        else:
            text = 'Unknown'
            
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị khung hình kết quả
    cv2.imshow('Face Recognition', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()