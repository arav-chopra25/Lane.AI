import cv2
import numpy as np
import os

# === UTILITY FUNCTIONS ===
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold=50, high_threshold=150):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is None or len(lines) == 0:
        return line_image

    for line in lines:
        if line is None or len(line) != 4:
            continue
        x1, y1, x2, y2 = line
        try:
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 10)
        except Exception:
            continue
    return line_image



def average_slope_intercept(image, lines):
    if lines is None:
        return []

    left_fit, right_fit = [], []
    for line in lines:
        if line is None:
            continue
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0:
            continue  # avoid division by zero
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    line_coords = []
    for fit in [left_fit, right_fit]:
        if len(fit) > 0:
            avg_fit = np.average(fit, axis=0)
            coords = make_coordinates(image, avg_fit)
            if coords is not None:
                line_coords.append(coords)

    return line_coords



def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    if slope == 0 or np.isinf(slope) or np.isnan(slope):
        return None  # skip broken line
    
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    
    x1 = (y1 - intercept) / slope
    x2 = (y2 - intercept) / slope

    if np.isinf(x1) or np.isinf(x2) or np.isnan(x1) or np.isnan(x2):
        return None  # invalid coords

    return np.array([int(x1), int(y1), int(x2), int(y2)])


# === PROCESS SINGLE IMAGE ===
def process_image(image):
    gray = grayscale(image)
    blur = gaussian_blur(gray)
    edges = canny(blur)
    cropped = region_of_interest(edges)
    lines = cv2.HoughLinesP(cropped, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(image, lines)
    line_image = display_lines(image, averaged_lines)
    combo = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combo

# === MAIN ===
if __name__ == "__main__":
    os.makedirs("output_images", exist_ok=True)
    os.makedirs("output_videos", exist_ok=True)

    print("📸 Processing test_images...")
    for filename in os.listdir("test_images"):
        if filename.endswith((".jpg", ".png")):
            path = os.path.join("test_images", filename)
            img = cv2.imread(path)
            if img is None:
                print(f"⚠️ Skipping {filename}, can't read")
                continue
            result = process_image(img)
            out_path = os.path.join("output_images", filename)
            cv2.imwrite(out_path, result)
            print(f"✅ Saved {out_path}")

    print("\n🎥 Processing test_videos...")
    for filename in os.listdir("test_videos"):
        if filename.endswith((".mp4", ".avi", ".mov")):
            path = os.path.join("test_videos", filename)
            cap = cv2.VideoCapture(path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = os.path.join("output_videos", filename)
            out = None
            print(f"▶️  Processing {filename}...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed = process_image(frame)
                if out is None:
                    height, width, _ = processed.shape
                    out = cv2.VideoWriter(out_path, fourcc, 20.0, (width, height))
                out.write(processed)

            cap.release()
            if out:
                out.release()
            print(f"✅ Saved {out_path}")

    print("\n🎯 All done! Check 'output_images' and 'output_videos'.")
