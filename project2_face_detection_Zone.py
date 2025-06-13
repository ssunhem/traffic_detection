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
        
        # Zone definition
        self.zone_points = [(0, 320), (1280, 320), (1280, 720), (0, 720)]
        self.zone_polygon = np.array(self.zone_points, dtype=np.int32)

        # Load the face detection model
        self.faceProto = r"D:\FD_FA\.venv1\model\opencv_face_detector.pbtxt"    # Path to the model files
        self.faceModel = r"D:\FD_FA\.venv1\model\opencv_face_detector_uint8.pb" # Path to the model files
        self.faceNet = cv2.dnn.readNet(self.faceModel, self.faceProto)

        self.face_records = []
        self.start_time = time.time()
        self.frame_count = 0

    def start_camera(self):  # Start the camera stream
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

    def highlightFace(self, frame, conf_threshold=0.8):  # Detect faces in the frame
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

    def stream_camera(self):  # Stream + fill zone + process only faces inside it
        while self.is_streaming:
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to retrieve frame.")
                break

            start_time = time.time()
            frame_resized = cv2.resize(frame, (1280, 720))
            resultImg, faceBoxes = self.highlightFace(frame_resized)

            # Create overlay and fill the zone with color
            overlay = resultImg.copy()
            cv2.fillPoly(overlay, [self.zone_polygon], color=(255, 255, 255))
            alpha = 0.01  # Transparency of the color (0.0–1.0)
            cv2.addWeighted(overlay, alpha, resultImg, 1 - alpha, 0, resultImg)

            # Draw zone boundary
            cv2.polylines(resultImg, [self.zone_polygon], isClosed=True, color=(255, 255, 255), thickness=2)

            # Filter boxes that are inside the zone
            filtered_boxes = []
            for x1, y1, x2, y2 in faceBoxes:
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                if cv2.pointPolygonTest(self.zone_polygon, (cx, cy), False) >= 0:
                    filtered_boxes.append((x1, y1, x2, y2))
                else:
                    cv2.rectangle(resultImg, (x1, y1), (x2, y2), (100,100,100), 1)

            # Overlay text (time, face count, FPS)
            now = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            cv2.putText(resultImg, now, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(resultImg, f"In Zone: {len(filtered_boxes)}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(resultImg, f"Analyzed: {len(self.data_records)}", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            fps = 1.0 / (time.time() - start_time) if time.time() > start_time else 0
            cv2.putText(resultImg, f"FPS: {fps:.2f}", (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Analyze only faces within the zone
            def analyze_face(box):
                x1, y1, x2, y2 = box
                crop = frame_resized[max(0, y1-self.padding):min(y2+self.padding, frame_resized.shape[0]),
                                    max(0, x1-self.padding):min(x2+self.padding, frame_resized.shape[1])]
                try:
                    face_resized = cv2.resize(crop, (50,100))
                    analysis = DeepFace.analyze(crop, actions=['age','gender','race'], enforce_detection=False)
                    age = analysis[0]['age']
                    gender = analysis[0]['dominant_gender']
                    race = analysis[0]['race']
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    _, buf = cv2.imencode('.jpg', face_resized)
                    self.data_records.append([ts, gender, age, race, buf.tobytes()])
                    cv2.putText(resultImg, f'{gender}, {age}, {race}', (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                except Exception as e:
                    print(f"Analysis error for box {box}: {e}")

            threads = []
            for fb in filtered_boxes:
                t = Thread(target=analyze_face, args=(fb,))
                t.daemon = True
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

            cv2.imshow(self.window_name, resultImg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_camera()
                break

    def stop_camera(self):  # Stop the camera stream
        if self.is_streaming:
            self.is_streaming = False
            if self.cap is not None:
                self.cap.release()
                cv2.destroyWindow(self.window_name)
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.export_data()

    def export_data(self):  # Save the analyzed data to an Excel file
        if not self.data_records:
            print("No data to save.")
            return

        current_time = datetime.now().strftime('%d%m%Y_%H%M%S')
        excel_path = os.path.join(self.output_folder, f"result_{current_time}.xlsx")

        workbook = xlsxwriter.Workbook(excel_path)
        worksheet = workbook.add_worksheet()

        # Set column width
        for col in range(9):  # Total 9 columns (0-8)
            worksheet.set_column(col, col, 12)  # Adjust width to fit content

        # Add headers with a new column "Day"
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

        row = 1  # Start from row 1 (row 0 is the header)

        for record in self.data_records:
            timestamp, gender, age, race, img_data = record

            # Convert Timestamp to datetime object and determine "Day"
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

            # Write data to the table
            worksheet.write_row(row, 0, [timestamp, day_type, gender, age, age_group, generation, dominant_race, race_percentage])

            # Convert binary image data to an image
            image_stream = BytesIO(img_data)
            image = Image.open(image_stream)

            # Resize image to fit the cell (80x80)
            image_resized = image.resize((80, 80))

            # Save to a temporary file
            temp_image_path = os.path.join(self.output_folder, f"temp_image_{row}.png")
            image_resized.save(temp_image_path)

            # Insert the image into the cell (H column = index 8)
            worksheet.insert_image(row, 8, temp_image_path, {
                'x_scale': 1.0,
                'y_scale': 1.0,
                'x_offset': 2,
                'y_offset': 2,
                'object_position': 1  # Image moves and is deleted when the cell is deleted
            })

            # Create column "J" named "check" with value 0
            worksheet.write(row, 9, 0)  # Write value 0 in the "check" column

            # Set row height to 80
            worksheet.set_row(row, 80)

            row += 1  # Move to the next row

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
