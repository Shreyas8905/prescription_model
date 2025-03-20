import cv2
import pytesseract
from PIL import Image
from opencv3 import CV
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
