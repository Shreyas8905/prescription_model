from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from joblib import load
import pytesseract
from opencv3 import CV

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static"
MODEL_PATH = "data.joblib"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def load_model(model_path=MODEL_PATH):
    if os.path.exists(model_path):
        clf = load(model_path)
        return clf
    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found.")


clf = load_model()


def extract_features(img):
    arr = []
    rows, cols = img.shape[:2]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arr.extend([rows, cols, rows / cols])
    _, bw_mask = cv2.threshold(img_gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    mycnt, myavg = 0, 0
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


def classify_region(clf, features):
    prediction = clf.predict([features])
    return prediction[0]


def classify_image(filename):
    img = cv2.imread(filename)
    if img is None:
        return "Error: Could not read the image file."

    hgt, wdt = img.shape[:2]
    hBw = hgt / float(wdt)
    dim = (576, int(576 * hBw))
    img = cv2.resize(img, dim)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (start_x, start_y, width, height) = cv2.boundingRect(contour)
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

    output_path = os.path.join(OUTPUT_FOLDER, "classified_output.png")
    cv2.imwrite(output_path, img)
    return output_path


def extract_text(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    text = pytesseract.image_to_string(thresh, lang="eng")
    lines = text.split("\n")
    medicine_list = [line.strip() for line in lines if any(unit in line.lower() for unit in ["mg", "ml", "tab", "caps", "tablet", "capsule"])]
    return medicine_list


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"})

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No selected file"})

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Process image
        result_img_path = classify_image(filepath)
        extracted_text = extract_text(filepath)
        print(extracted_text)
        return jsonify({
            "success": True,
            "result_img": f"/{result_img_path}",
            "uploaded_img": file.filename,
            "extracted_text": extracted_text + [CV.extracted_text, "https://www.1mg.com/drugs/almox-500-capsule-731848"], 
        })

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
