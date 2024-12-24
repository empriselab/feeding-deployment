"""
__________.__                                      _______               
\______   \  |   ____  ______ __________   _____   \      \ _____ ___  __
 |    |  _/  |  /  _ \/  ___//  ___/  _ \ /     \  /   |   \\__  \\  \/ /
 |    |   \  |_(  <_> )___ \ \___ (  <_> )  Y Y  \/    |    \/ __ \\   / 
 |______  /____/\____/____  >____  >____/|__|_|  /\____|__  (____  /\_/  
        \/                \/     \/            \/         \/     \/      

Copyright (c) 2024 Interactions Lab
License: MIT
Authors: Anthony Song and Nathan Dennler, Cornell University & University of Southern California
Project Page: https://github.com/interaction-lab/BlossomNav.git

This script contains an GUI that allows recording video footage from the Pi Zero 2 and saving to local computer

"""

import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import os

class VideoStreamApp:
    def __init__(self, root, rtsp_link):
        self.root = root
        self.rtsp_link = rtsp_link
        self.root.title("RTSP Video Stream")
        self.video_frame = tk.Label(root)
        self.video_frame.pack()
        self.start_button = tk.Button(root, text="Start", command=self.start_saving)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = tk.Button(root, text="Stop", command=self.stop_saving)
        self.stop_button.pack(side=tk.RIGHT)
        self.saving = False
        self.cap = cv2.VideoCapture(self.rtsp_link)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Unable to open video stream")
            self.root.quit()
        self.frame_counter = 0
        self.image_dir = "saved_images"
        os.makedirs(self.image_dir, exist_ok=True)
        self.current_frame = None
        self.lock = threading.Lock()
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.start()
        self.update_stream()  # Start the stream update loop

    def start_saving(self):
        if not self.saving:
            self.saving = True
            self.frame_counter = 0
            messagebox.showinfo("Info", "Started saving frames")
        

    def stop_saving(self):
        if self.saving:
            self.saving = False
            messagebox.showinfo("Info", "Stopped saving frames")
            self.compile_video()

    def capture_frames(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame

    def update_stream(self):
        with self.lock:
            if self.current_frame is not None:
                frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)
                if self.saving:
                    image_path = os.path.join(self.image_dir, f"frame_{self.frame_counter:05d}.png")
                    cv2.imwrite(image_path, self.current_frame)
                    self.frame_counter += 1
        self.root.after(100, self.update_stream)  # Schedule the next update in 100 ms

    def compile_video(self):
        filename = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        if not filename:
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10.0
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
        for i in range(self.frame_counter):
            image_path = os.path.join(self.image_dir, f"frame_{i:05d}.png")
            frame = cv2.imread(image_path)
            out.write(frame)
            os.remove(image_path)
        out.release()
        messagebox.showinfo("Info", "Video compiled successfully")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoStreamApp(root, 'http://192.168.1.5:8083/')
    root.mainloop()