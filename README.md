# 👨‍💻 Cấu Trúc Pipeline Nhận Dạng Khuôn Mặt

Sơ đồ này mô tả một quy trình (pipeline) hiện đại và mạnh mẽ để nhận dạng khuôn mặt người, sử dụng sự kết hợp của ba công nghệ cốt lõi: **MTCNN**, **FaceNet**, và **SVM**.

## 1. ⚙️ Quy Trình Hoạt Động (The Pipeline)

Quy trình được chia thành ba giai đoạn xử lý chính:

| Giai Đoạn | Công Cụ | Chức Năng Cốt Lõi | Đầu Ra |
| :--- | :--- | :--- | :--- |
| **I** | **MTCNN** | Phát hiện và Căn chỉnh khuôn mặt. | Khuôn mặt đã được cắt và chuẩn hóa. |
| **II** | **FaceNet** | Trích xuất các đặc trưng độc nhất (Embedding). | Vector số (128 chiều) đại diện cho khuôn mặt. |
| **III** | **SVM** | Phân loại và Nhận dạng tên người. | Tên của người được nhận dạng. |

---

## 2. 🟢 Giai Đoạn I: Phát Hiện và Căn Chỉnh (MTCNN)

* **Công cụ:** **MTCNN** (Multi-task Cascaded Convolutional Neural Network).
* **Mục đích:** Đảm bảo rằng chỉ có khuôn mặt được xử lý, loại bỏ nhiễu từ nền (background), và chuẩn hóa vị trí khuôn mặt.
* **Cách thức:**
    * Nhận **Ảnh Đầu Vào (Input Image)**.
    * Phát hiện tất cả vị trí khuôn mặt trong ảnh, kể cả khi khuôn mặt nghiêng hoặc xoay.
    * Xác định 5 điểm mốc quan trọng (mắt, mũi, miệng) để **căn chỉnh** khuôn mặt về một kích thước và góc độ chuẩn.

## 3. 🟡 Giai Đoạn II: Trích Xuất Đặc Trưng (FaceNet)

* **Công cụ:** **FaceNet**.
* **Mục đích:** Chuyển đổi khuôn mặt thành một chuỗi số (vector) có thể so sánh được.
* **Cách thức:**
    * Lấy khuôn mặt đã được MTCNN căn chỉnh.
    * Sử dụng mạng nơ-ron để tạo ra một **Face Embedding** (Mã hóa Khuôn mặt) duy nhất.
    * **Đặc điểm của Embedding:** Khoảng cách giữa các embedding của cùng một người là **rất nhỏ**, trong khi khoảng cách giữa các embedding của những người khác nhau là **rất lớn**.

## 4. 🔴 Giai Đoạn III: Phân Loại và Nhận Dạng (SVM)

* **Công cụ:** **SVM** (Support Vector Machine).
* **Mục đích:** Học cách phân loại các Face Embedding vào đúng tên người.
* **Cách thức:**
    * **Đào tạo:** SVM được đào tạo trên các Face Embedding của những người đã biết (ví dụ: Ivan, Ana) và học cách tạo ra ranh giới phân loại giữa họ.
    * **Dự đoán:** Khi một embedding mới được cung cấp (từ khuôn mặt chưa biết), SVM xác định embedding đó rơi vào vùng phân loại nào và trả về tên của người đó.

## 5. 🎯 Kết Quả

Quy trình kết thúc bằng việc trả lời câu hỏi: **"Who is this?"** (Người này là ai?), cung cấp thông tin nhận dạng về **Person** (Người) được phát hiện trong ảnh.
<img src="C:\Users\khoal\OneDrive\Tài liệu\GitHub\FACE_RECOGNITION\resouce\video_test_face_reg.gif" alt="Demo tính năng đăng nhập" width="500"/>
