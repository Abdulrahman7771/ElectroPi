#import easyocr
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import pytesseract
pd.set_option('display.max_columns', None)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread("ID.jpg")
gray = cv2.imread("ID.jpg",0)
rgb = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

#reader = easyocr.Reader(['ar', 'en'],gpu=True)
#result = reader.readtext(rgb)

print(pytesseract.image_to_string(rgb,lang="eng"))

#imgDataFrame = pd.DataFrame(result,columns=["Box","Text","Percentage"])
#print(imgDataFrame)

plt.imshow(rgb)










































