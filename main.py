import cv2
import numpy as np
from joblib import dump, load
import os
import cv2
import pytesseract
from PIL import Image
from opencv3 import CV
def extract_features(img):
    arr = []
    rows, cols = img.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arr.extend([rows, cols, rows / cols])
    _, bw_mask = cv2.threshold(img_gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mycnt = 0
    myavg = 0
    for xx in range(cols):
        mycnt = 0
        for yy in range(rows):
            if bw_mask[yy, xx] == 0:
                mycnt += 1
        myavg += (mycnt * 1.0) / rows
    myavg /= cols
    arr.append(myavg)
    change = 0
    for xx in range(rows):
        mycnt = 0
        for yy in range(cols - 1):
            if bw_mask[xx, yy] != bw_mask[xx, yy + 1]:
                mycnt += 1
        change += (mycnt * 1.0) / cols
    change /= rows
    arr.append(change)
    return arr
def train_model(training_data, labels, model_path="data.joblib"):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(training_data, labels)
    dump(clf, model_path)
    print(f"Model trained and saved to '{model_path}'")
def load_model(model_path="model.joblib"):
    if os.path.exists(model_path):
        clf = load(model_path)
        print(f"Model loaded from '{model_path}'")
        return clf
    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
def classify_region(clf, features):
    prediction = clf.predict([features])
    return prediction[0]
def classify_image(filename, model_path="data.joblib"):
    clf = load_model(model_path)
    img = cv2.imread(filename)
    if img is None:
        print("Error: Could not read the image file.")
        return
    hgt, wdt = img.shape[:2]
    hBw = hgt / float(wdt)
    dim = (576, int(576 * hBw))
    fram = img.copy()
    img = cv2.resize(img, dim)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    linek = np.zeros((11, 11), dtype=np.uint8)
    linek[5, ...] = 1
    x = cv2.morphologyEx(gray, cv2.MORPH_OPEN, linek, iterations=1)
    gray -= x
    kernel = np.ones((5, 5), np.uint8)
    _, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray = cv2.dilate(gray, kernel, iterations=1)
    contours, _ = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for x in range(len(contours)):
        (start_x, start_y, width, height) = cv2.boundingRect(contours[x])
        mymat = img[start_y:start_y + height, start_x:start_x + width]
        features = extract_features(mymat)
        label = classify_region(clf, features)
        color_map = {
            "Printed_extended": (255, 0, 0),  
            "Handwritten_extended": (0, 255, 0),  
            "Mixed_extended": (0, 0, 255),  
            "Other_extended": (0, 255, 255),
        }
        cv2.rectangle(img, (start_x, start_y), (start_x + width, start_y + height), color_map.get(label, (255, 255, 255)), 2)
    output_path = "classified_output.png"
    cv2.imwrite(output_path, img)
    print(f"Classification completed. Results saved to '{output_path}'")
image_path = "image.png"
classify_image(image_path)
image_path = "image.png"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
text = pytesseract.image_to_string(thresh, lang='eng')
lines = text.split("\n")
medicine_list = []
for line in lines:
    if any(unit in line.lower() for unit in ["mg", "ml", "tab", "caps", "tablet", "capsule"]):
        medicine_list.append(line.strip())
print(line)
print("Extracted Text: ")
print(CV.extracted_text)