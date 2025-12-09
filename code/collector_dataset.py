import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk  # Cần cài đặt Pillow: pip install Pillow
import cv2
import os
import time

# Biến toàn cục
selected_path = ""
cap = None  # Đối tượng VideoCapture
webcam_active = False # Trạng thái webcam
image_count = 0 # Đếm số ảnh đã chụp
current_folder_full_path = "" # Lưu đường dẫn thư mục hiện tại để chụp ảnh

def select_directory():
    """Mở hộp thoại chọn thư mục và lưu đường dẫn đã chọn."""
    global selected_path
    path = filedialog.askdirectory()
    if path:
        selected_path = path
        path_label.config(text=f"Đường dẫn đã chọn: {selected_path}")
    else:
        selected_path = ""
        path_label.config(text="Chưa chọn đường dẫn")

def start_capture_process():
    """Bắt đầu quá trình tạo thư mục và mở webcam."""
    global cap, webcam_active, image_count, current_folder_full_path

    folder_name = entry_name.get().strip()

    if not folder_name:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập tên cho thư mục!")
        return

    if not selected_path:
        messagebox.showwarning("Cảnh báo", "Vui lòng chọn đường dẫn để lưu thư mục!")
        return

    # Tạo đường dẫn đầy đủ cho thư mục mới
    current_folder_full_path = os.path.join(selected_path, folder_name)

    # Tạo thư mục
    try:
        if not os.path.exists(current_folder_full_path):
            os.makedirs(current_folder_full_path)
            messagebox.showinfo("Thông báo", f"Thư mục '{folder_name}' đã được tạo thành công tại:\n{current_folder_full_path}")
        else:
            messagebox.showwarning("Cảnh báo", f"Thư mục '{folder_name}' đã tồn tại tại '{selected_path}'.\nẢnh sẽ được lưu vào thư mục này.")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể tạo thư mục: {e}")
        return

    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở webcam. Vui lòng kiểm tra lại thiết bị.")
        return

    webcam_active = True
    image_count = 0
    # Bắt đầu hiển thị và chụp ảnh
    capture_frame()
    # Vô hiệu hóa các nút/ô nhập liệu không cần thiết trong quá trình chụp
    button_select_path.config(state=tk.DISABLED)
    entry_name.config(state=tk.DISABLED)
    button_start.config(state=tk.DISABLED)
    button_stop.config(state=tk.NORMAL) # Kích hoạt nút dừng

def capture_frame():
    """Đọc frame từ webcam, hiển thị trên Tkinter và chụp ảnh."""
    global cap, webcam_active, image_count, current_folder_full_path

    if webcam_active and cap is not None:
        ret, frame = cap.read()
        if ret:
            # Chuyển đổi màu sắc từ BGR (OpenCV) sang RGB (Pillow)
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Thay đổi kích thước frame cho phù hợp với hiển thị trên UI (tùy chọn)
            # Giữ tỉ lệ khung hình nếu muốn
            h, w, _ = cv2_image.shape
            ratio = w / h
            new_width = 480 # Chiều rộng mong muốn
            new_height = int(new_width / ratio)
            if new_height > 360: # Giới hạn chiều cao để không quá lớn
                new_height = 360
                new_width = int(new_height * ratio)

            cv2_image = cv2.resize(cv2_image, (new_width, new_height))

            img = Image.fromarray(cv2_image)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)

            # Chụp ảnh nếu chưa đủ 20 ảnh và đã qua 0.5 giây kể từ lần chụp trước
            if image_count < 20 and (not hasattr(capture_frame, 'last_capture_time') or (time.time() - capture_frame.last_capture_time) >= 0.5):
                image_filename = os.path.join(current_folder_full_path, f"{os.path.basename(current_folder_full_path)}_{image_count}.jpg")
                cv2.imwrite(image_filename, frame) # Lưu ảnh gốc (không bị resize)
                print(f"Đã chụp và lưu: {image_filename}")
                image_count += 1
                capture_frame.last_capture_time = time.time()
                
            if image_count >= 20:
                stop_capture_process()
                messagebox.showinfo("Hoàn tất", f"Đã chụp {image_count} ảnh và lưu vào thư mục '{os.path.basename(current_folder_full_path)}' tại:\n{current_folder_full_path}")
                return # Thoát khỏi hàm để không lên lịch cuộc gọi tiếp theo

            # Lên lịch cuộc gọi lại hàm này sau 10ms (tạo hiệu ứng video)
            video_label.after(10, capture_frame)
        else:
            stop_capture_process()
            messagebox.showerror("Lỗi", "Không thể đọc frame từ webcam.")
    elif cap is not None: # Nếu webcam không còn active nhưng cap vẫn tồn tại (đã dừng)
        stop_capture_process()

def stop_capture_process():
    """Dừng quá trình chụp ảnh và giải phóng webcam."""
    global cap, webcam_active
    webcam_active = False
    if cap:
        cap.release()
        cap = None
    video_label.config(image='') # Xóa hình ảnh trên label
    video_label.config(text="Webcam đã dừng") # Hiển thị thông báo
    # Kích hoạt lại các nút/ô nhập liệu
    button_select_path.config(state=tk.NORMAL)
    entry_name.config(state=tk.NORMAL)
    button_start.config(state=tk.NORMAL)
    button_stop.config(state=tk.DISABLED) # Vô hiệu hóa nút dừng

# --- Thiết lập giao diện Tkinter ---
root = tk.Tk()
root.title("Tạo thư mục và Chụp ảnh từ Webcam")
root.geometry("700x600") # Kích thước cửa sổ lớn hơn

# --- Phần chọn đường dẫn ---
path_frame = tk.Frame(root)
path_frame.pack(pady=10)

label_path_selection = tk.Label(path_frame, text="1. Chọn đường dẫn lưu thư mục:")
label_path_selection.pack(side=tk.LEFT, padx=5)

button_select_path = tk.Button(path_frame, text="Chọn đường dẫn", command=select_directory)
button_select_path.pack(side=tk.LEFT, padx=5)

path_label = tk.Label(root, text="Chưa chọn đường dẫn", fg="blue")
path_label.pack(pady=5)

# --- Phần nhập tên thư mục ---
name_frame = tk.Frame(root)
name_frame.pack(pady=10)

label_name = tk.Label(name_frame, text="2. Nhập tên cho thư mục:")
label_name.pack(side=tk.LEFT, padx=5)

entry_name = tk.Entry(name_frame, width=30)
entry_name.pack(side=tk.LEFT, padx=5)

# --- Nút để bắt đầu và dừng quá trình ---
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

button_start = tk.Button(button_frame, text="3. Bắt đầu (Tạo & Mở Webcam)", command=start_capture_process, bg="green", fg="white", font=("Arial", 10, "bold"))
button_start.pack(side=tk.LEFT, padx=10)

button_stop = tk.Button(button_frame, text="Dừng Webcam & Chụp ảnh", command=stop_capture_process, bg="red", fg="white", font=("Arial", 10, "bold"), state=tk.DISABLED)
button_stop.pack(side=tk.LEFT, padx=10)


# --- Phần hiển thị Webcam ---
video_label = tk.Label(root, text="Webcam sẽ hiển thị ở đây", width=480, height=360, bg="black", fg="white")
video_label.pack(pady=15)

root.mainloop()

# Đảm bảo webcam được giải phóng khi đóng cửa sổ
if cap:
    cap.release()
    cv2.destroyAllWindows()