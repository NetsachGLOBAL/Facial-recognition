import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
import face_recognition
import cv2
import os
import datetime
from PIL import Image, ImageTk

# Directory with known face images
known_faces_dir = r"d:\Facial Recignition\known_people"
log_file = "recognition_log.txt"

# Load known faces and their names
known_face_encodings = []
known_face_names = []

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(known_faces_dir):
        image_path = os.path.join(known_faces_dir, filename)
        if os.path.isfile(image_path):
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(os.path.splitext(filename)[0])

# Load known faces initially
load_known_faces()

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Tool")
        
        self.video_capture = cv2.VideoCapture(0)
        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.start_button = tk.Button(root, text="Start Video", command=self.start_video)
        self.start_button.pack(side=tk.LEFT)
        
        self.stop_button = tk.Button(root, text="Stop Video", command=self.stop_video)
        self.stop_button.pack(side=tk.LEFT)
        
        self.add_face_button = tk.Button(root, text="Add New Face", command=self.add_new_face)
        self.add_face_button.pack(side=tk.LEFT)
        
        self.running = False

    def start_video(self):
        self.running = True
        self.update_video()

    def stop_video(self):
        self.running = False

    def update_video(self):
        if self.running:
            ret, frame = self.video_capture.read()
            if not ret:
                return
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                name = "Unknown"
                confidence = 0

                if True in matches:
                    best_match_index = matches.index(True)
                    name = known_face_names[best_match_index]
                    confidence = (1 - face_distances[best_match_index]) * 100

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f}%)", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                with open(log_file, "a") as log:
                    log.write(f"{datetime.datetime.now()} - {name} ({confidence:.2f}%)\n")

            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.root.after(10, self.update_video)

    def add_new_face(self):
        new_name = simpledialog.askstring("Input", "Enter the name of the new face:")
        if new_name:
            ret, frame = self.video_capture.read()
            if ret:
                new_face_path = f"{known_faces_dir}/{new_name}.jpg"
                cv2.imwrite(new_face_path, frame)
                load_known_faces()
                messagebox.showinfo("Success", f"Added {new_name} to the known faces.")
            else:
                messagebox.showerror("Error", "Failed to capture image. Please try again.")

    def __del__(self):
        if self.video_capture.isOpened():
            self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
