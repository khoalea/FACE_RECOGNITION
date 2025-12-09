# Import các thư viện cần thiết
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.model_selection import train_test_split # Để chia dữ liệu huấn luyện và kiểm tra
from sklearn.metrics import accuracy_score # Để đánh giá độ chính xác
import matplotlib.pyplot as plt # Thư viện để vẽ đồ thị (Cần cài đặt: pip install matplotlib)
import numpy as np # Thư viện để xử lý mảng và ma trận


# ---------------------------------------------------------------------------------
# User Library: Dataset
# ---------------------------------------------------------------------------------
import dataset as ds

# ---------------------------------------------------------------------------------
# Phần thực thi chính
# ---------------------------------------------------------------------------------

print("--- Minh họa SVM: Phân loại Embeddings và Biểu diễn trực quan ---")


print("\n-----------------------------------------------------")
print("                  DỮ LIỆU ĐẦU VÀO MẪU")
print("-----------------------------------------------------")
print("Đây là các 'vector đặc trưng' (embeddings) giả định 2 chiều:")
print(ds.X)
print("\nVà đây là 'nhãn' (tên người) tương ứng của chúng:")
print(ds.y)
print(f"Tổng số mẫu dữ liệu: {len(ds.X)}")

# 2. Chia dữ liệu thành tập huấn luyện (training) và tập kiểm tra (testing)
X_train, X_test, y_train, y_test = train_test_split(ds.X, ds.y, test_size=0.3, random_state=42)

print(f"\nSố mẫu dùng để huấn luyện SVM: {len(X_train)}")
print(f"Số mẫu dùng để kiểm tra SVM: {len(X_test)}")

# 3. Khởi tạo và huấn luyện mô hình SVM
model_svm = SVC(kernel='linear')
print("\n-----------------------------------------------------")
print("            QUÁ TRÌNH HUẤN LUYỆN SVM")
print("-----------------------------------------------------")
print("Bắt đầu huấn luyện mô hình SVM với dữ liệu đã cho...")
model_svm.fit(X_train, y_train)
print("Huấn luyện SVM hoàn tất! Mô hình đã 'học' cách phân loại.")

# 4. Dự đoán trên tập kiểm tra
y_pred = model_svm.predict(X_test)

print("\n-----------------------------------------------------")
print("              KẾT QUẢ DỰ ĐOÁN CỦA SVM")
print("-----------------------------------------------------")
print("Nhãn mà SVM dự đoán trên tập kiểm tra (y_pred):")
print(y_pred)
print("Nhãn thực tế của các mẫu trong tập kiểm tra (y_test):")
print(y_test)

# 5. Đánh giá độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"\nĐộ chính xác của mô hình SVM trên tập kiểm tra: {accuracy * 100:.2f}%")

# 6. Minh họa dự đoán một mẫu mới
new_embedding_person_A = np.array([[0.13, 0.23]])
new_embedding_person_B = np.array([[1.01, 1.11]])
new_embedding_unknown = np.array([[5.0, 5.0]])

print("\n-----------------------------------------------------")
print("          DỰ ĐOÁN CHO CÁC MẪU MỚI (CHƯA BIẾT)")
print("-----------------------------------------------------")
prediction_A = model_svm.predict(new_embedding_person_A)
print(f"Embedding mới {new_embedding_person_A[0]} được dự đoán là: Người {prediction_A[0]}")

prediction_B = model_svm.predict(new_embedding_person_B)
print(f"Embedding mới {new_embedding_person_B[0]} được dự đoán là: Người {prediction_B[0]}")

prediction_unknown = model_svm.predict(new_embedding_unknown)
print(f"Embedding mới {new_embedding_unknown[0]} được dự đoán là: Người {prediction_unknown[0]} (sẽ là nhóm gần nhất đã học)")

print("\nHoàn thành minh họa SVM về mặt số học.")

# ---------------------------------------------------------------------------------
# KHỐI HIỂN THỊ ĐỒ HỌA (DISPLAY BLOCK) - DÀNH CHO MÔI TRƯỜNG CỤC BỘ
# Học sinh cần chạy code này trên máy tính cá nhân để thấy biểu đồ
# ---------------------------------------------------------------------------------
print("\n-----------------------------------------------------")
print("          KHỐI HIỂN THỊ ĐỒ HỌA SVM (OPTIONAL)")
print("-----------------------------------------------------")
print("Để trực quan hóa cách SVM phân loại, hãy chạy phần code dưới đây TRÊN MÁY CÁ NHÂN của bạn.")
print("Bạn sẽ thấy một biểu đồ với các điểm dữ liệu và đường phân chia của SVM.")

try:
    # Bước 1: Chuẩn bị dữ liệu và huấn luyện lại mô hình (để vẽ được)
    # Gộp X_train và y_train để vẽ
    unique_labels = np.unique(y_train)
    colors = ['blue', 'red', 'green'] # Màu sắc cho các lớp A, B, C (hoặc nhiều hơn)
    label_map = {label: i for i, label in enumerate(unique_labels)}

    # Tạo lưới để vẽ đường biên giới
    x_min, x_max = ds.X[:, 0].min() - 0.5, ds.X[:, 0].max() + 0.5
    y_min, y_max = ds.X[:, 1].min() - 0.5, ds.X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Dự đoán trên lưới để vẽ đường biên giới
    Z = model_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([label_map[label] for label in Z]).reshape(xx.shape) # Chuyển nhãn về số để vẽ

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu) # Vùng phân loại

    # Vẽ các điểm dữ liệu huấn luyện
    for i, label in enumerate(unique_labels):
        idx = np.where(y_train == label)
        plt.scatter(X_train[idx, 0], X_train[idx, 1], c=colors[i], label=f'Người {label}',
                    edgecolor='k', s=80, marker='o')

    # Vẽ các support vectors (điểm quan trọng SVM dùng để vẽ đường biên)
    # Chỉ hoạt động với kernel='linear' và SVM đã huấn luyện
    if hasattr(model_svm, 'support_vectors_'):
        plt.scatter(model_svm.support_vectors_[:, 0], model_svm.support_vectors_[:, 1], s=150,
                    facecolors='none', edgecolors='black', label='Support Vectors')

    plt.xlabel('Đặc trưng 1 (Embedding Dimension 1)')
    plt.ylabel('Đặc trưng 2 (Embedding Dimension 2)')
    plt.title('Minh họa Hoạt động của SVM: Phân chia dữ liệu')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend()
    plt.grid(True)
    plt.show() # Lệnh này sẽ mở cửa sổ biểu đồ trên máy cá nhân

except Exception as e:
    print(f"\nKhông thể hiển thị biểu đồ đồ họa trực tiếp trong môi trường này.")
    print(f"Vui lòng chạy file code trên máy tính cá nhân và đảm bảo đã cài đặt 'matplotlib' (pip install matplotlib) để xem biểu đồ.")
    print(f"Lỗi: {e}")

print("\n-----------------------------------------------------")
print("          KẾT THÚC MINH HỌA SVM")
print("-----------------------------------------------------")