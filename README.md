# 🚗 Lane Detection AI

This project is a real-time lane detection system built using Python and OpenCV. It allows users to upload images or videos of roads with visible lanes. The AI then processes each frame, detects lane boundaries using computer vision techniques, and overlays them onto the original media.

Perfect for learning the basics of computer vision, real-time image processing, and video analysis for autonomous driving or ADAS systems.

---

## ✨ Features

- Upload and process **test images** or **full road videos**
- Automatic detection of left and right lane markings
- Processes videos frame-by-frame and saves the output with detected lanes overlayed
- Works on straight and slightly curved lanes
- Efficient masking and line detection with Canny + Hough Transform
- Automatic output saving for images and videos

---

## 🧠 How It Works

1. Convert the input frame (image/video) to grayscale  
2. Apply Gaussian Blur to reduce noise  
3. Use Canny Edge Detection to identify edges  
4. Mask region of interest (road area)  
5. Detect lines using Hough Transform  
6. Average out left and right lane slopes  
7. Draw the final detected lanes on the original media

---

## 🧑‍💻 Tech Stack

- **Language:** Python 3.x  
- **Libraries:** OpenCV, NumPy, OS

---

## 📁 Folder Structure

```plaintext
│
├── lane_detection.py          # Main script
├── test_images/               # Folder for input images
├── test_videos/               # Folder for input videos
├── output_images/             # Output images with lanes
├── output_videos/             # Output videos with lanes
└── README.md                  # Project documentation
