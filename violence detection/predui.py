import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import os

# Load the trained model
model = load_model('violence_detection_model_bestnew.h5')

# Initialize variables
violence_counter = 0
violence_detected = False

# Function to predict the class of each frame
def predict_frame_class(frame, model):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_resized = np.array(frame_resized, dtype=np.float32) / 255.0
    frame_resized = np.expand_dims(frame_resized, axis=0)  # Add batch dimension
    prediction = model.predict(frame_resized)

    # Predict the class (0 = Violence, 1 = Non-Violence)
    if np.argmax(prediction) == 0:
        return "Violence"
    else:
        return "Non-Violence"

# Function to display video with predictions
def display_video_with_classification(video_path, model):
    global violence_counter, violence_detected
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Predict the class for the current frame
        frame_class = predict_frame_class(frame, model)

        if frame_class == "Violence":
            violence_counter += 1
        else:
            violence_counter = 0

        # Trigger action if violence detected for 4 seconds (assuming 30 FPS)
        if violence_counter >= 40 and not violence_detected:
            violence_detected = True

        # Add text to the frame indicating the classification
        cv2.putText(frame, f"Class: {frame_class}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame with the classification
        cv2.imshow("Video with Classification", frame)

        # Exit video display on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to browse for a video file
def browse_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.avi *.mp4 *.mov")])
    if video_path:
        messagebox.showinfo("Selected File", f"Selected Video: {video_path}")

# Function to start detection
def start_detection():
    if not video_path:
        messagebox.showerror("Error", "Please select a video file first.")
    else:
        display_video_with_classification(video_path, model)

# Create the main Tkinter window
root = tk.Tk()
root.title("CYPHER CAM MASTER")
root.geometry("800x600")

# Add background image
bg_image = Image.open("bg.jpg")  # Replace with your background image file path
bg_image = bg_image.resize((800, 600), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)
background_label = tk.Label(root, image=bg_photo)
background_label.place(relwidth=1, relheight=1)

# Add header
header = tk.Label(root, text="CYPHER CAM MASTER", font=("Helvetica", 24, "bold"), bg="black", fg="white")
header.pack(pady=20)

# Add browse button
browse_button = tk.Button(root, text="Browse Video File", font=("Helvetica", 16), command=browse_video, bg="blue", fg="white")
browse_button.pack(pady=20)

# Add detect button
detect_button = tk.Button(root, text="Detect", font=("Helvetica", 16), command=start_detection, bg="green", fg="white")
detect_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
