import numpy as np
from ultralytics import YOLO
import cv2
from datetime import datetime, timedelta
import os
from openpyxl import Workbook
from openpyxl.drawing.image import Image
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkcalendar import Calendar
import customtkinter
from threading import Thread

# กำหนดเกณฑ์ความมั่นใจขั้นต่ำสำหรับการตรวจจับ
CONFIDENCE_THRESHOLD = 0.5
# โฟลเดอร์สำหรับบันทึกภาพและไฟล์ Excel ที่ได้จากการตรวจจับ
OUTPUT_FOLDER = r"D:\output_results"

# ตรวจสอบและสร้างโฟลเดอร์ผลลัพธ์หากยังไม่มี
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
# สร้างโฟลเดอร์อีกครั้งโดยไม่เกิดข้อผิดพลาดหากมีอยู่แล้ว
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ขนาดการปรับรีไซส์เฟรมวิดีโอเพื่อการประมวลผล
RESIZE_WIDTH, RESIZE_HEIGHT = 1280, 720

# กำหนดพิกัดของโซนรูปหลายเหลี่ยมที่สนใจตรวจจับ
zone_points = [(0, 0), (1280, 0), (1280, 720), (0, 720)]
zone_polygon = np.array(zone_points, dtype=np.int32)

# ค้นหาไฟล์โมเดล YOLO โดยเดินสำรวจไดเรกทอรีหลัก
base_dir = r"D:\AI_Project\proj_code_nt\code\Project1_body_detection"
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "model_n_80_20.pt":  # ตรวจสอบชื่อไฟล์โมเดล
            model_path = os.path.join(root, file)
            break
# โหลดโมเดล YOLO เมื่อพบไฟล์แล้ว
model = YOLO(model_path)
print(f"Model loaded from: {model_path}")

# ฟังก์ชันเปิดหน้าต่างเลือกไฟล์วิดีโอ
def import_video():
    global video_path
    video_path = filedialog.askopenfilename(
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )
    status_label.config(text=f"Loaded: {video_path}")
    print(f"Selected video: {video_path}")

# ฟังก์ชันให้ผู้ใช้เลือกวัน-เวลาเริ่มต้นการตรวจจับ
def pick_datetime():
    def set_datetime():
        global start_time_dt
        selected_date = cal.get_date()  # ดึงค่าวันจากปฏิทิน
        selected_time = f"{hour_var.get()}:{minute_var.get()}"  # ดึงค่าเวลาจาก Spinbox
        # รวมวันและเวลาเป็น datetime object
        start_time_dt = datetime.strptime(f"{selected_date} {selected_time}", "%m/%d/%y %H:%M")
        datetime_window.destroy()  # ปิดหน้าต่างเลือกวัน-เวลา
        print(f"Start time set to: {start_time_dt}")
        status_label.config(text=f"{video_path} | time set to: {start_time_dt}")
    # สร้างหน้าต่างย่อยสำหรับเลือกวัน-เวลา
    datetime_window = tk.Toplevel(root)
    datetime_window.title("Select Start Time")
    cal = Calendar(datetime_window, selectmode='day', year=2025, month=2, day=27)  # ปฏิทินเลือกวัน
    cal.pack(pady=10)
    time_frame = tk.Frame(datetime_window)  # กรอบจัดวาง Spinbox
    time_frame.pack(pady=5)
    hour_var = tk.StringVar(value="00")
    minute_var = tk.StringVar(value="00")
    hour_spinbox = ttk.Spinbox(time_frame, from_=0, to=23, textvariable=hour_var, width=5, format="%02.0f")
    minute_spinbox = ttk.Spinbox(time_frame, from_=0, to=59, textvariable=minute_var, width=5, format="%02.0f")
    hour_spinbox.pack(side=tk.LEFT)
    tk.Label(time_frame, text=":").pack(side=tk.LEFT)
    minute_spinbox.pack(side=tk.LEFT)
    tk.Button(datetime_window, text="Set Time", command=set_datetime).pack(pady=5)

# ฟังก์ชันเริ่มต้นการตรวจจับแบบรันใน Thread แยก
def start_detection():
    global detection_running
    if not video_path:
        status_label.config(text="Please select a video file first.")
        return
    if not start_time_dt:
        status_label.config(text="Please select a start time first.")
        return
    detection_running = True
    Thread(target=process_video).start()  # เริ่มกระบวนการตรวจจับใน Thread ใหม่

# ฟังก์ชันหยุดการตรวจจับ
def stop_detection():
    global detection_running
    detection_running = False

# ตรวจสอบว่าจุดอยู่ในโซนหรือไม่
def is_point_in_zone(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# แปลงชื่อคลาสเป็นเพศ
def get_gender(class_name):
    female_classes = ["fe1", "fe2", "fe3", "fe4", "FeM"]
    male_classes = ["ma1", "ma2", "ma3", "ma4"]
    if class_name in female_classes:
        return "Female"
    elif class_name in male_classes:
        return "Male"
    else:
        return "Unknown"

# แปลงชื่อคลาสเป็นช่วงอายุ
def get_age_range(class_name):
    if class_name in ["fe1", "ma1"]:
        return "child"
    elif class_name in ["fe2", "ma2"]:
        return "youth"
    elif class_name in ["FeM", "fe3", "ma3"]:
        return "adult"
    elif class_name in ["fe4", "ma4"]:
        return "elderly"
    else:
        return "Unknown"

# แปลงชื่อคลาสเป็นศาสนา
def get_religion(class_name):
    otherreligion_classes = ["fe1", "fe2", "fe3", "fe4", "ma1", "ma2", "ma3", "ma4"]
    islam_classes = ["FeM"]
    if class_name in otherreligion_classes:
        return "Other Religion"
    elif class_name in islam_classes:
        return "Islam"
    else:
        return "Unknown"

# ฟังก์ชันหลักสำหรับประมวลผลวิดีโอและตรวจจับ
def process_video():
    global detection_running
    cap = cv2.VideoCapture(video_path)
    # เตรียมไฟล์ Excel และหัวคอลัมน์
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Detection Results"
    sheet.append(["Detection Time", "Class", "Confidence", "Gender", "Age Range", "Religion", "ID", "Image Path"])
    recorded_track_ids = set()  # เก็บ track ID ที่บันทึกแล้ว
    start_time_actual = datetime.now()  # เวลาอ้างอิงสำหรับคำนวณเวลา
    while detection_running:
        ret, frame = cap.read()
        if not ret:
            status_label.config(text="Detection successful.")
            break
        # ปรับขนาดเฟรมและวาดโซนที่สนใจ
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        cv2.polylines(frame, [zone_polygon], isClosed=True, color=(0, 255, 0), thickness=1)
        # คำนวณเวลาการตรวจจับโดยอิง start_time_dt
        elapsed_seconds = (datetime.now() - start_time_actual).total_seconds()
        detection_time = (start_time_dt + timedelta(seconds=elapsed_seconds)).strftime("%Y-%m-%d %H:%M:%S")
        # เรียกใช้ YOLO สำหรับ tracking
        results = model.track(source=frame, persist=True, conf=CONFIDENCE_THRESHOLD, tracker="bytetrack.yaml")
        if not results:
            continue
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # พิกัดขอบเขตกล่อง
                    conf = box.conf[0]  # คะแนนความมั่นใจ
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    track_id = int(box.id[0])
                except:
                    continue
                # คำนวณพิกัดศูนย์กลางกล่อง
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                # ข้ามหากนอกโซนที่กำหนด
                if not is_point_in_zone(centroid, zone_polygon):
                    continue
                # บันทึกครั้งแรกของแต่ละ track
                if track_id not in recorded_track_ids:
                    recorded_track_ids.add(track_id)
                    cropped_image = frame[y1:y2, x1:x2]
                    # บันทึกภาพและตั้งชื่อไฟล์ให้เหมาะสม
                    image_filename = f"{detection_time.replace(':', '-')}_{class_name}_{track_id}.jpg"
                    image_path = os.path.join(OUTPUT_FOLDER, image_filename)
                    cv2.imwrite(image_path, cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    # ดึงข้อมูลเพศ อายุ และศาสนา
                    gender = get_gender(class_name)
                    age_range = get_age_range(class_name)
                    religion = get_religion(class_name)
                    # เขียนข้อมูลลง Excel
                    sheet.append([detection_time, class_name, f"{conf:.2f}", gender, age_range, religion, track_id, image_path])
                    # ใส่ภาพใน Excel
                    img = Image(image_path)
                    img.height = 200
                    img.width = 100
                    sheet.add_image(img, f"H{sheet.max_row}")
                    # วาดจุดศูนย์กลางบนเฟรมเพื่อแสดงตำแหน่ง
                    cv2.circle(frame, centroid, 5, (0, 255, 255), -1)
        # แสดงผลลัพธ์บนหน้าต่าง
        cv2.imshow("YOLO Video Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # เพิ่มสูตรนับจำนวนทั้งหมดใน Excel และบันทึกไฟล์
    sheet["J1"] = "Total Count"
    sheet["J2"] = "=COUNTA(A2:A100)"
    workbook.save(os.path.join(OUTPUT_FOLDER, f"output_{start_time_dt}_model.xlsx"))

# ---------------- การตั้งค่า GUI ----------------
root = tk.Tk()
root.title("YOLO Video Detection")
# ประกาศตัวแปร global
video_path = ""
detection_running = False
start_time_dt = None
# กำหนดขนาดหน้าต่างหลัก
root.minsize(500, 500)
root.geometry("500x500")
# สร้างหัวเรื่องบน GUI
h1 = tk.Label(root, text="Video Detection", font=("Arial", 20))
h1.pack(ipady=15, fill="x")
# ป้ายสถานะสำหรับแจ้งการโหลดไฟล์และสถานะ
status_label = tk.Label(root, text="No video loaded")
status_label.pack(fill="x")
# ปุ่มเลือกวิดีโอ
btn_import = customtkinter.CTkButton(
    master=root,
    text="Import Video",
    text_color="#ffffff",
    hover=True,
    hover_color="#262626",
    border_width=4,
    corner_radius=6,
    border_color="#000000",
    bg_color="#ffffff",
    fg_color="#004d1a",
    command=import_video
)
btn_import.place(x=180, y=100)
# ปุ่มเลือกวัน-เวลาเริ่มต้น
btn_pick_time = customtkinter.CTkButton(
    master=root,
    text="Pick Start Time",
    text_color="#ffffff",
    hover=True,
    hover_color="#262626",
    border_width=4,
    corner_radius=6,
    border_color="#000000",
    bg_color="#ffffff",
    fg_color="#004d1a",
    command=pick_datetime
)
btn_pick_time.place(x=180, y=200)
# ปุ่มเริ่มตรวจจับ
btn_start = customtkinter.CTkButton(
    master=root,
    text="Start Detection",
    text_color="#ffffff",
    hover=True,
    hover_color="#262626",
    border_width=4,
    corner_radius=6,
    border_color="#000000",
    bg_color="#ffffff",
    fg_color="#004d1a",
    command=start_detection
)
btn_start.place(x=180, y=300)
# ปุ่มหยุดตรวจจับ
btn_stop = customtkinter.CTkButton(
    master=root,
    text="Stop",
    text_color="#ffffff",
    hover=True,
    hover_color="#262626",
    border_width=4,
    corner_radius=6,
    border_color="#000000",
    bg_color="#ffffff",
    fg_color="#004d1a",
    command=stop_detection
)
btn_stop.place(x=180, y=400)
# เริ่มต้นลูปหลักของ Tkinter
root.mainloop()