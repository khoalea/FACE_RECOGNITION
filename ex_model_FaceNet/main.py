# Import các thư viện cần thiết
import numpy as np # Thư viện NumPy để làm việc với mảng số
from PIL import Image # Thư viện Pillow để đọc và xử lý ảnh
from keras_facenet import FaceNet # Thư viện FaceNet để trích xuất embeddings
import matplotlib.pyplot as plt # Thư viện để vẽ đồ thị
from mpl_toolkits.mplot3d import Axes3D # Để vẽ đồ thị 3D
from sklearn.decomposition import PCA # Để giảm chiều dữ liệu
import os # Để làm việc với đường dẫn file

# ---------------------------------------------------------------------------------
# HƯỚNG DẪN CHUẨN BỊ TRƯỚC KHI CHẠY CODE:
# 1. Cài đặt các thư viện:
#    Mở Terminal/Command Prompt và chạy các lệnh sau:
#    pip install numpy Pillow keras_facenet tensorflow matplotlib scikit-learn
#    (Lưu ý: keras_facenet sẽ tự động tải mô hình FaceNet đã pre-trained khi khởi tạo)
#
# 2. Chuẩn bị ảnh khuôn mặt đầu vào:
#    Để FaceNet hoạt động hiệu quả nhất, ảnh đầu vào nên là ảnh của MỘT KHUÔN MẶT DUY NHẤT,
#    đã được CẮT và CĂN CHỈNH (aligned) về kích thước chuẩn 160x160 pixels.
#    Bạn có thể sử dụng output từ bước MTCNN (nếu có), hoặc tự cắt/căn chỉnh các ảnh mẫu.
#
#    Tạo một thư mục 'sample_faces' trong cùng thư mục với script này.
#    Bên trong 'sample_faces', tạo các thư mục con với tên của từng người
#    (ví dụ: 'An', 'Binh', 'Cuong') và đặt các ảnh khuôn mặt của họ vào đó.
#    Mỗi người nên có ít nhất 2-3 ảnh để thấy được sự nhóm lại.
#
#    Cấu trúc thư mục sẽ trông như sau:
#    your_project_folder/
#    ├── facenet_standalone_demo.py
#    └── sample_faces/
#        ├── An/
#        │   ├── an_1.jpg
#        │   ├── an_2.png
#        │   └── ...
#        ├── Binh/
#        │   ├── binh_1.jpg
#        │   ├── binh_2.png
#        │   └── ...
#        └── Cuong/
#            ├── cuong_1.jpg
#            ├── cuong_2.png
#            └── ...
#
#    Bạn có thể dùng các ảnh placeholder để thử nghiệm, ví dụ:
#    https://placehold.co/160x160/FF0000/FFFFFF?text=An1
#    https://placehold.co/160x160/FF0000/FFFFFF?text=An2
#    https://placehold.co/160x160/00FF00/FFFFFF?text=Binh1
#    https://placehold.co/160x160/00FF00/FFFFFF?text=Binh2
#    https://placehold.co/0000FF/FFFFFF?text=Cuong1
#    https://placehold.co/0000FF/FFFFFF?text=Cuong2
# ---------------------------------------------------------------------------------

# Khởi tạo mô hình FaceNet
# Khi dòng này được chạy lần đầu, thư viện keras_facenet sẽ tự động tải
# mô hình FaceNet đã được huấn luyện sẵn (pre-trained model).
# Đảm bảo bạn có kết nối internet khi chạy lần đầu tiên.
embedder = FaceNet()
print("FaceNet model đã được khởi tạo và sẵn sàng sử dụng.")

# Hàm để tải và tiền xử lý ảnh khuôn mặt cho FaceNet
def load_and_preprocess_face_image(image_path, required_size=(160, 160)):
    """
    Tải ảnh khuôn mặt, chuyển đổi sang RGB và resize về kích thước yêu cầu.
    Đầu vào:
    - image_path: Đường dẫn đến file ảnh khuôn mặt.
    - required_size: Kích thước mà FaceNet mong đợi (mặc định 160x160).
    Đầu ra:
    - Mảng numpy chứa dữ liệu pixel của ảnh khuôn mặt đã được xử lý, hoặc None nếu có lỗi.
    """
    try:
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize(required_size)
        face_array = np.asarray(image)
        return face_array
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file ảnh tại đường dẫn: {image_path}")
        return None
    except Exception as e:
        print(f"LỖI: Xảy ra lỗi khi tải hoặc xử lý ảnh '{image_path}': {e}")
        return None

# Hàm chính để lấy embedding của khuôn mặt
def get_face_embedding(image_path):
    """
    Tải ảnh khuôn mặt, tiền xử lý và trích xuất vector đặc trưng (embedding) bằng FaceNet.
    Đầu vào:
    - image_path: Đường dẫn đến file ảnh khuôn mặt (đã được cắt và căn chỉnh).
    Đầu ra:
    - Một mảng numpy 128 chiều, là vector đặc trưng của khuôn mặt, hoặc None nếu có lỗi.
    """
    face_pixels = load_and_preprocess_face_image(image_path)
    if face_pixels is None:
        return None

    try:
        # embedder.embeddings() mong đợi một danh sách các mảng numpy của khuôn mặt.
        embedding = embedder.embeddings([face_pixels])[0]
        return embedding
    except Exception as e:
        print(f"LỖI: Không thể trích xuất embedding từ '{image_path}'. Lỗi: {e}")
        return None

# Hàm để tải và xử lý nhiều ảnh từ thư mục
def load_and_process_multiple_faces(base_dir):
    """
    Tải tất cả ảnh khuôn mặt từ các thư mục con, trích xuất embeddings và lưu nhãn.
    Đầu vào:
    - base_dir: Đường dẫn tới thư mục gốc chứa các thư mục con của từng người (e.g., 'sample_faces/').
    Đầu ra:
    - all_embeddings: Danh sách các vector đặc trưng (128 chiều) của tất cả các khuôn mặt.
    - all_labels: Danh sách các nhãn (tên người) tương ứng với mỗi embedding.
    """
    all_embeddings = []
    all_labels = []
    print(f"\nĐang tải và xử lý ảnh từ thư mục: {base_dir}")

    for person_name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_name)
        if os.path.isdir(person_dir):
            print(f"  Đang xử lý ảnh cho: {person_name}")
            for image_filename in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_filename)
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    embedding = get_face_embedding(image_path)
                    if embedding is not None:
                        all_embeddings.append(embedding)
                        all_labels.append(person_name)
                        print(f"    -> Đã trích xuất embedding cho '{image_filename}' của {person_name}")
    return np.array(all_embeddings), np.array(all_labels)

# ---------------------------------------------------------------------------------
# PHẦN THỰC THI CHÍNH CỦA CHƯƠNG TRÌNH
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- BẮT ĐẦU MINH HỌA FACENET ĐỘC LẬP VỚI NHIỀU ẢNH ---")

    # Định nghĩa đường dẫn tới thư mục chứa các ảnh khuôn mặt mẫu
    SAMPLE_FACES_DIR = 'dataset'

    # Tải và xử lý tất cả các ảnh, lấy embeddings và nhãn
    embeddings, labels = load_and_process_multiple_faces(SAMPLE_FACES_DIR)

    if len(embeddings) == 0:
        print("\nKhông tìm thấy ảnh hoặc không thể trích xuất embedding nào.")
        print("Vui lòng kiểm tra cấu trúc thư mục 'sample_faces' và các file ảnh.")
        print("--- KẾT THÚC MINH HỌA FACENET ĐỘC LẬP ---")
        exit()

    print(f"\nTổng số embeddings đã trích xuất: {len(embeddings)}")
    print(f"Kích thước của tập embeddings: {embeddings.shape}")

    # ---------------------------------------------------------------------------------
    # MINH HỌA TRỰC QUAN BẰNG ĐỒ THỊ 3D (SỬ DỤNG PCA)
    # ---------------------------------------------------------------------------------
    print("\n--- BẮT ĐẦU MINH HỌA TRỰC QUAN 3D CÁC EMBEDDINGS ---")
    print("Các embeddings có 128 chiều, chúng ta sẽ dùng PCA để giảm xuống 3 chiều để vẽ đồ thị.")

    # Bước 1: Giảm chiều dữ liệu bằng PCA (Principal Component Analysis)
    # PCA giúp chúng ta tìm ra 3 "hướng" quan trọng nhất trong dữ liệu 128 chiều
    # để biểu diễn chúng trong không gian 3D mà vẫn giữ được nhiều thông tin nhất.
    pca = PCA(n_components=3) # Giảm xuống 3 chiều
    reduced_embeddings = pca.fit_transform(embeddings)
    print(f"Kích thước embeddings sau khi giảm chiều bằng PCA: {reduced_embeddings.shape}")

    # Bước 2: Chuẩn bị dữ liệu để vẽ đồ thị
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Gán màu sắc và marker cho từng người
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels)) # Lấy 10 màu khác nhau
    markers = ['o', '^', 's', 'P', 'X', 'D', 'v', '<', '>', 'h'] # Các loại marker khác nhau

    # Bước 3: Vẽ các điểm dữ liệu lên đồ thị 3D
    print("Đang vẽ đồ thị 3D các embeddings...")
    for i, label in enumerate(unique_labels):
        # Lấy các embeddings thuộc về người hiện tại
        indices = np.where(labels == label)
        ax.scatter(reduced_embeddings[indices, 0],
                   reduced_embeddings[indices, 1],
                   reduced_embeddings[indices, 2],
                   color=colors(i),
                   marker=markers[i % len(markers)], # Đảm bảo có đủ marker
                   label=f'Người: {label}',
                   s=60) # Kích thước điểm

    # Đặt tiêu đề và nhãn cho các trục
    ax.set_xlabel('Thành phần chính 1 (PC1)')
    ax.set_ylabel('Thành phần chính 2 (PC2)')
    ax.set_zlabel('Thành phần chính 3 (PC3)')
    ax.set_title('Biểu đồ 3D các Face Embeddings (sau PCA)')
    ax.legend()
    ax.grid(True)

    # Hiển thị đồ thị
    print("\nĐồ thị 3D sẽ xuất hiện trong một cửa sổ mới.")
    print("Quan sát các cụm điểm: Các điểm của cùng một người sẽ có xu hướng nhóm lại gần nhau.")
    plt.show()

    print("\n--- KẾT THÚC MINH HỌA FACENET ĐỘC LẬP ---")
