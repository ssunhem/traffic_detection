import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

points = []
image = None
canvas = None
photo = None

def open_image():
    global image, photo, canvas, points
    
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return
    
    image = cv2.imread(file_path)
    image = cv2.resize(image, (1280, 720))  
    points.clear()  
    display_image()

def display_image():
    global image, photo, canvas
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    photo = ImageTk.PhotoImage(image_pil)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    
    coord_label.config(text="Coordinates: " + ", ".join([f"({x}, {y})" for x, y in points]))

def click_event(event):
    global image, points
    
    x, y = event.x, event.y
    points.append((x, y))
    
    coordinates = f"({x}, {y})"
    cv2.putText(image, coordinates, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        

    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    if len(points) > 1:
        cv2.line(image, points[-2], points[-1], (255, 0, 0), 2)
    
    display_image()


#GUI
root = tk.Tk()
root.title("Image Click Coordinates")

btn_open = tk.Button(root, text="Open Image", command=open_image)
btn_open.pack()

canvas = tk.Canvas(root, width=1280, height=720)
canvas.pack()
canvas.bind("<Button-1>", click_event)

coord_label = tk.Label(root, text="Coordinates: None")
coord_label.pack()

root.mainloop()
