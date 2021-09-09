import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import PySimpleGUI as sg
sg.theme('Light Blue 2')

layout = [[sg.Text('Enter Image file ')],
          [sg.Text('File ', size=(8, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Text('File1 ', size=(8, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Submit(), sg.Cancel()]]

window = sg.Window('Image ', layout)

event, values = window.read()
window.close()
print(f'You clicked {event}')
print(f'You chose filenames {values[0]}')
print(f'You chose filenames {values[1]}')
text_input = values[0] 
text_input1=values[1]
# Open the image files. 
img1_color = cv2.imread(text_input)          # queryImage
img2_color = cv2.imread(text_input1) # trainImage

# Convert to grayscale. 
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) 
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY) 
height, width = img2.shape 
def chi2_distance(histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])
		# return the chi-squared distance
		return d
  
# Create ORB detector with 5000 features. 
orb_detector = cv2.ORB_create() 
  
# Find keypoints and descriptors. 
# The first arg is the image, second arg is the mask 
#  (which is not reqiured in this case). 
kp1, d1 = orb_detector.detectAndCompute(img1, None) 
kp2, d2 = orb_detector.detectAndCompute(img2, None) 
  
# Match features between the two images. 
# We create a Brute Force matcher with  
# Hamming distance as measurement mode.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
# Match the two sets of descriptors. 
matches = matcher.match(d1, d2) 
# Sort matches on the basis of their Hamming distance. 
matches.sort(key = lambda x: x.distance) 
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches, None, flags=2)
#plt.imshow(img3),plt.show()
cv2.imwrite("img3.jpg",img3)
#cv2.imshow("asjhd",img3)
#plt.show()

# Take the top 90 % matches forward. 
#matches = matches[:int(len(matches)*90)] 
no_of_matches = len(matches) 
# Save the output. 
dist =len(matches) / (max(len(d1), len(d2)))
d = chi2_distance(d1, d2)
#print(dist)
import PySimpleGUI as sg
import PySimpleGUI as sg
import io
from PIL import Image
sg.theme('Light Blue 2')
layout = [
    [sg.Output(key='-OUT-', size=(50, 10))],
        [sg.Image(key="-IMAGE-")],
]
window = sg.Window("Image Viewer", layout,finalize=True,auto_close=True)
#window = sg.Window('Image shape Analysis', layout, element_justification='center').finalize()
window['-OUT-'].TKOut.output.config(wrap='word') # set Output element word wrapping

print(dist)
print(img1_color.shape)
print(img2_color.shape)
if(dist<0.20):
    print("same Laptops/Objects")
else:
    print("Different Laptops/objects")
image = Image.open("img3.jpg")
image.thumbnail((800, 800))
bio = io.BytesIO()
image.save(bio, format="PNG")
window["-IMAGE-"].update(data=bio.getvalue())
while True:
    win, ev, val = sg.read_all_windows()
    if ev == sg.WIN_CLOSED:
        win.close()
        break
window.close()