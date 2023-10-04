import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import streamlit as st

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

file = st.file_uploader(
    "Upload your data file",
    accept_multiple_files=False,
    key=st.session_state["file_uploader_key"],
)

if file:
    st.session_state["uploaded_files"] = file

if file is not None:
        img = cv2.imread(file.name)
st.image(file)
img = cv2.imread('ID.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((2,2))


ret, thresh = cv2.threshold(gray
                            , 127, 255
                            ,  cv2.THRESH_BINARY)


laplacian = cv2.Laplacian(thresh, -1,ksize=7)
OLaplacian =laplacian
laplacian = laplacian[:laplacian.shape[0]-int(laplacian.shape[1]/6), int(laplacian.shape[1]/3):laplacian.shape[1]]


rgb = cv2.cvtColor(laplacian, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.show()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
result = pytesseract.image_to_string(laplacian,lang="ara")
print(result)

laplacian = OLaplacian[OLaplacian.shape[0]-int(OLaplacian.shape[0]/3):OLaplacian.shape[0],int(OLaplacian.shape[1]/3):OLaplacian.shape[1]]
rgb = cv2.cvtColor(laplacian, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.show()
numres = pytesseract.image_to_string(laplacian
                                     ,lang="aranumberLayer2")
print(numres)

i=0
for word in result.split("\n"):
    if(word != ""):
        print(i,": ",word)
        i=i+1



























