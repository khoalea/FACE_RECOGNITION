# Import các thư viện cần thiết
from mtcnn.mtcnn import MTCNN # Thư viện MTCNN để phát hiện khuôn mặt
import cv2 # Thư viện OpenCV để đọc, xử lý và hiển thị ảnh
import matplotlib.pyplot as plt # Thư viện để hiển thị ảnh
# import tensorflow as tf # Thư viện TensorFlow để kiểm tra GPU
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# ---------------------------------------------------------------------------------
# Chuẩn bị: Đảm bảo có một file ảnh tên 'sample_face.jpg' trong cùng thư mục với script này
# hoặc thay đổi đường dẫn tới ảnh của bạn.
# Bạn có thể tải một ảnh bất kỳ có khuôn mặt rõ ràng.
# ---------------------------------------------------------------------------------

# 1. Khởi tạo bộ phát hiện khuôn mặt MTCNN
# MTCNN() sẽ tải model đã được huấn luyện sẵn.
detector = MTCNN()
print("MTCNN detector đã được khởi tạo.")
# 2. Đọc ảnh đầu vào
# image_path = 'sample_face.jpg' # Tên file ảnh mẫu
image_path = 'dataset//trump//Screenshot 2025-06-19 111948.png'
try:
    image = cv2.imread(image_path)
    # OpenCV đọc ảnh dưới dạng BGR, Matplotlib hiển thị RGB nên cần chuyển đổi
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Đã đọc ảnh: {image_path}")
except Exception as e:
    print(f"Lỗi khi đọc ảnh: {e}. Vui lòng kiểm tra đường dẫn và tên file ảnh.")
    exit()

# 3. Phát hiện khuôn mặt trong ảnh
# detector.detect_faces() sẽ trả về một danh sách các khuôn mặt được phát hiện.
# Mỗi khuôn mặt là một dictionary chứa 'box' (vị trí) và 'keypoints' (các điểm mốc).
faces = detector.detect_faces(image_rgb)
print(f"Tìm thấy {len(faces)} khuôn mặt trong ảnh.")

# 4. Vẽ bounding box và các điểm mốc lên ảnh
# Chúng ta sẽ vẽ trực tiếp lên ảnh RGB để hiển thị bằng matplotlib
draw_image = image_rgb.copy()

for face in faces:
    # Lấy thông tin về vị trí khuôn mặt (bounding box)
    x, y, width, height = face['box']
    # Vẽ hình chữ nhật quanh khuôn mặt (màu xanh lá)
    cv2.rectangle(draw_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    print(f"  Khuôn mặt tại: X={x}, Y={y}, Width={width}, Height={height}")

    # Lấy thông tin về các điểm mốc (keypoints)
    keypoints = face['keypoints']
    print("  Điểm mốc:")
    for key, value in keypoints.items():
        # Vẽ chấm tròn tại mỗi điểm mốc (màu đỏ)
        cv2.circle(draw_image, value, 2, (255, 0, 0), -1)
        print(f"    {key}: {value}")

# 5. Hiển thị ảnh kết quả
plt.imshow(draw_image)
plt.title(f'Khuôn mặt được phát hiện và điểm mốc trong ảnh: {image_path}')
plt.axis('off') # Tắt trục tọa độ
plt.show()

print("\nHoàn thành minh họa MTCNN.")