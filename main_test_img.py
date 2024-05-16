import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

thres = 0.52  # Threshold to detect objects

# Function to process image
def process_image(img):
    # Detect objects in the image
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        num_cars = 0
        num_buses = 0
        num_motorcycles = 0
        num_persons = 0
        
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Ensure classId is within the valid range
            if 0 <= classId < len(classNames):
                if classNames[classId - 1].lower() == 'car':
                    num_cars += 1
                elif classNames[classId - 1].lower() == 'bus':
                    num_buses += 1
                elif classNames[classId - 1].lower() == 'motorcycle':
                    num_motorcycles += 1
                elif classNames[classId - 1].lower() == 'person':
                    num_persons += 1
                
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                print(f"Invalid classId: {classId}")

        # Check for traffic environment
        if num_cars > 2 and num_buses > 0 and num_motorcycles > 0 and num_persons > 2:
            cv2.putText(img, "Traffic ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(img, "Normal Environment", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return img

# Open the video capture with camera index 0 (you can change it to 2 if needed)
cap = cv2.VideoCapture(0)

# Set a higher resolution
cap.set(3, 1920)
cap.set(4, 1080)
cap.set(10, 70)

# Load class names from coco.names file
classNames = []
classFile = 'names.txt'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the pre-trained model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Create the main GUI window
root = tk.Tk()
root.title("Object Detection App")

# Create a label to display the video stream
label = ttk.Label(root)
label.pack(padx=10, pady=10)

# Set a larger window size
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Output', 1280, 720)  # Set the desired window size

def on_key_press(event):
    # Check if the pressed key is 'q'
    if event.char == 'q':
        root.destroy()

# Bind the key press event to the on_key_press function
root.bind('<KeyPress>', on_key_press)

# Function to handle image processing and display
def process_and_display(img):
    # Resize the image to match the expected input size
    img = cv2.resize(img, (700, 500))
    img = process_image(img)

    # Convert the image to RGB format for tkinter
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))

    # Update the label with the new image
    label.img = img_tk
    label.config(image=img_tk)

# Function to handle camera capture
def capture_from_camera():
    success, img = cap.read()
    if success:
        process_and_display(img)
        root.after(10, capture_from_camera)

# Function to handle image selection from file dialog
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        img = cv2.imread(file_path)
        if img is not None:
            process_and_display(img)

# Create buttons for camera and image input
camera_button = ttk.Button(root, text="Use Camera", command=capture_from_camera)
camera_button.pack(side=tk.LEFT, padx=10, pady=10)

image_button = ttk.Button(root, text="Select Image", command=select_image)
image_button.pack(side=tk.RIGHT, padx=10, pady=10)

# Start the GUI main loop
root.mainloop()

# Release the capture object and close the window
cap.release()
cv2.destroyAllWindows()
