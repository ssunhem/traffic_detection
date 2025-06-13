import time
import cv2
import tkinter as tk
from tkinter import messagebox
from threading import Thread
from deepface import DeepFace

import os
import datetime
from datetime import datetime

from PIL import Image
from io import BytesIO
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
import numpy as np
import xlsxwriter
from tkinter import messagebox, filedialog


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Control")
        self.cap = None
        self.is_streaming = False
        self.faceNet = None
        self.frame_count = 0
        self.padding = 10
        self.data_records = []
        self.output_folder = "captured_faces"
        os.makedirs(self.output_folder, exist_ok=True)
        self.window_name = "Detection"

        root.minsize(550, 500)
        root.geometry("550x500")
        h1 = tk.Label(root, text="Customer Data Collection Systems", font=("Arial", 20, "bold"))
        h1.pack(ipady=25, fill="x")
        

        # Input selection: Webcam or Video
        self.input_type = tk.StringVar(value="webcam")  # Default is webcam

        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=10)

        webcam_radio = tk.Radiobutton(input_frame, text="Use Webcam", variable=self.input_type, value="webcam", font=("Arial", 12))
        video_radio = tk.Radiobutton(input_frame, text="Import Video File", variable=self.input_type, value="video", font=("Arial", 12))
        webcam_radio.pack(side=tk.LEFT, padx=10)
        video_radio.pack(side=tk.LEFT, padx=10)

        # Start and Stop buttons
        self.start_button = tk.Button(self.root, text="Start Camera", 
                                    command=self.start_camera, width=20, height=3, bg="green", fg="white", font=("Arial", 12, "bold"), relief="raised", bd=10)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Stop Camera",
                                    command=self.stop_camera, state=tk.DISABLED, width=20, height=3, bg="red", fg="white", font=("Arial", 12, "bold"),relief="raised", bd=10)
        self.stop_button.pack(pady=10)
        

        # Load the face detection model
        self.faceProto = r"D:\FD_FA\.venv1\model\opencv_face_detector.pbtxt"
        self.faceModel = r"D:\FD_FA\.venv1\model\opencv_face_detector_uint8.pb"
        self.faceNet = cv2.dnn.readNet(self.faceModel, self.faceProto)

        self.face_records = []
        self.start_time = time.time()
        self.frame_count = 0

    def start_camera(self):
        if not self.is_streaming:
            try:
                input_choice = self.input_type.get()
                if input_choice == "video":
                    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
                    if not file_path:
                        messagebox.showinfo("Cancelled", "No video file selected.")
                        return
                    self.cap = cv2.VideoCapture(file_path)
                else:
                    self.cap = cv2.VideoCapture(0)

                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Unable to open video source.")
                    return

                self.is_streaming = True
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)

                self.stream_thread = Thread(target=self.stream_camera)
                self.stream_thread.daemon = True
                self.stream_thread.start()
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

    def highlightFace(self, frame, conf_threshold=0.8):
        frameOpencvDnn = frame.copy()
        frameHeight, frameWidth = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        faceBoxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frameOpencvDnn, faceBoxes

    def stream_camera(self):
        while self.is_streaming:
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to retrieve frame.")
                break

            # Start time before processing the frame for real-time FPS calculation
            start_time = time.time()

            # # Process every 2nd frame to reduce load
            # self.frame_count += 1
            # if self.frame_count % 2 != 0:  # Process every 2 frame to reduce load
            #     continue

            # Resize frame for faster processing
            frame_resized = cv2.resize(frame, (1280, 720))

            # Display date and time on the top-left corner
            resultImg, faceBoxes = self.highlightFace(frame_resized)
            current_datetime = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            cv2.putText(resultImg, current_datetime, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display detection count
            detection_count_text = f"Detect : {len(faceBoxes)}"
            cv2.putText(resultImg, detection_count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display analyzed faces count
            analyzed_faces_count = len(self.data_records)
            analyzed_faces_text = f"Count : {analyzed_faces_count}"
            cv2.putText(resultImg, analyzed_faces_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Calculate real-time FPS
            end_time = time.time()
            time_per_frame = end_time - start_time
            if time_per_frame > 0:
                self.fps = 1 / time_per_frame
                fps_text = f"FPS : {self.fps:.2f}"
                cv2.putText(resultImg, fps_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                self.fps = 0

            # Face analysis function
            def analyze_face(faceBox):
                x1, y1, x2, y2 = faceBox
                face = frame_resized[max(0, y1 - self.padding):min(y2 + self.padding, frame_resized.shape[0] - 1),
                                    max(0, x1 - self.padding):min(x2 + self.padding, frame_resized.shape[1] - 1)]
                try:
                    face_resized = cv2.resize(face, (50, 100))
                    face = frame_resized[y1:y2, x1:x2]
                    analysis = DeepFace.analyze(face, actions=['age', 'gender', 'race'], enforce_detection=False)
                    age = analysis[0]['age']
                    gender = analysis[0]['dominant_gender']
                    race = analysis[0]['race']
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # Convert image to binary
                    _, buffer = cv2.imencode(".jpg", face_resized)
                    image_binary = buffer.tobytes()

                    self.data_records.append([timestamp, gender, age, race, image_binary])

                    cv2.putText(resultImg, f'{gender}, {age}, {race}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                except Exception as e:
                    print(f"Error analyzing face at frame {self.frame_count}, face box {faceBox}: {e}")

            # Remove duplicate faces
            faceBoxes = self.remove_duplicate_faces(faceBoxes)

            # Multithreading for faster face analysis
            threads = []
            for faceBox in faceBoxes:
                thread = Thread(target=analyze_face, args=(faceBox,))
                thread.start()
                threads.append(thread)

            # Join threads to ensure completion
            for thread in threads:
                thread.join()

            # Show the result image
            cv2.imshow(self.window_name, resultImg)

            # Stop the stream if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_camera()
                break

    def remove_duplicate_faces(self, faceBoxes, threshold=50):
        new_face_boxes = []
        for faceBox in faceBoxes:
            duplicate = False
            for saved_box in self.face_records:
                dist = self.compute_distance(faceBox, saved_box)
                if dist < threshold:
                    duplicate = True
                    break
            if not duplicate:
                new_face_boxes.append(faceBox)
                self.face_records.append(faceBox)
        return new_face_boxes

    def compute_distance(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        center1 = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        center2 = np.array([(x1_ + x2_) / 2, (y1_ + y2_) / 2])
        return np.linalg.norm(center1 - center2)

    def stop_camera(self):
        if self.is_streaming:
            self.is_streaming = False
            if self.cap is not None:
                self.cap.release()
                cv2.destroyWindow(self.window_name)
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.export_data()

    def export_data(self):
        if not self.data_records:
            print("No data to save.")
            return

        current_time = datetime.now().strftime('%d%m%Y_%H%M%S')
        excel_path = os.path.join(self.output_folder, f"result_{current_time}.xlsx")

        workbook = xlsxwriter.Workbook(excel_path)
        worksheet = workbook.add_worksheet()

        # ตั้งค่าความกว้างของทุกคอลัมน์เป็น 80
        for col in range(9):  # มีทั้งหมด 9 คอลัมน์ (0-8)
            worksheet.set_column(col, col, 12)  # 12 เป็นค่าที่เหมาะสมให้พอดีกับเนื้อหา

        # เพิ่มหัวตาราง พร้อมคอลัมน์ใหม่ "Day"
        headers = ["Timestamp", "Day", "Gender", "Age", "Age Group", "Generation", "Race", "Race_Percentage", "Image"]
        worksheet.write_row(0, 0, headers)

        def classify_age(age):
            if age <= 1:
                return "Infancy"
            elif 1 < age <= 9:
                return "Childhood"
            elif 10 <= age <= 19:
                return "Adolescence"
            elif 20 <= age <= 39:
                return "Early adulthood"
            elif 40 <= age <= 59:
                return "Middle adulthood"
            else:
                return "Older age"

        def classify_generation(age):
            birth_year = datetime.now().year - age
            if 1928 <= birth_year <= 1945:
                return "The Silent Generation"
            elif 1946 <= birth_year <= 1964:
                return "Baby Boomers"
            elif 1965 <= birth_year <= 1980:
                return "Generation X"
            elif 1981 <= birth_year <= 1996:
                return "Generation Y"
            elif 1997 <= birth_year <= 2012:
                return "Generation Z"
            elif 2013 <= birth_year <= 2024:
                return "Generation Alpha"
            elif birth_year >= 2025:
                return "Generation Beta"
            else:
                return "Unknown"

        row = 1  # เริ่มจากแถวที่ 1 (แถว 0 เป็น header)

        for record in self.data_records:
            timestamp, gender, age, race, img_data = record

            # แปลง Timestamp เป็น datetime object และกำหนดค่า "Day"
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")  
            day_type = "Weekend" if dt.weekday() >= 5 else "Weekdays"

            age_group = classify_age(age)
            generation = classify_generation(age)

            # Classify race and get dominant race and percentage
            if isinstance(race, dict):
                dominant_race = max(race, key=race.get)
                race_percentage = race[dominant_race]
            else:
                dominant_race = str(race)
                race_percentage = None

            # เขียนข้อมูลลงตาราง
            worksheet.write_row(row, 0, [timestamp, day_type, gender, age, age_group, generation, dominant_race, race_percentage])

            # แปลง Binary image data เป็นรูป
            image_stream = BytesIO(img_data)
            image = Image.open(image_stream)

            # Resize image ให้พอดีกับเซลล์ (80x80)
            image_resized = image.resize((80, 80))

            # Save ลงไฟล์ชั่วคราว
            temp_image_path = os.path.join(self.output_folder, f"temp_image_{row}.png")
            image_resized.save(temp_image_path)

            # แทรกรูปเข้าไปในเซลล์ (H column = index 8)
            worksheet.insert_image(row, 8, temp_image_path, {
                'x_scale': 1.0,
                'y_scale': 1.0,
                'x_offset': 2,
                'y_offset': 2,
                'object_position': 1  # รูปขยับและถูกลบเมื่อเซลล์ถูกลบ
            })

            #create column "J" name "check" value is 0
            worksheet.write(row, 9, 0)  # Write value 0 in the "check" column


            # ตั้งค่าความสูงของแถวให้เป็น 80
            worksheet.set_row(row, 80)

            row += 1  # ไปแถวถัดไป

        workbook.close()
        print(f"Data saved to {excel_path}")
        # Display success message on GUI
        success_label = tk.Label(self.root, text=f"Data saved to : \n {excel_path}", fg="green", font=("Arial", 14, "bold"))
        success_label.pack(pady=20)

        # Automatically remove the label after 15 seconds
        self.root.after(15000, success_label.destroy)

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
