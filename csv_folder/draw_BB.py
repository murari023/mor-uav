import cv2
import pandas as pd
import sys

if len(sys.argv) < 3:
    print("usage: python draw_BB.py <image_file in jpg> <csv_file in txt>")
    exit()
 
image_file = sys.argv[1]
csv_file = sys.argv[2]

# Read RGB image 
img = cv2.imread(image_file)
  
# Output img with window name as 'image' 
#cv2.imshow('image', img)  
  
df_BB = pd.read_csv(csv_file, header=None) 

# creating new Image object 
#img = Image.new("RGB", (w, h)) 
for index, row in df_BB.iterrows():
    img_name = row[0]

    start_point = (row[1], row[2]) 
    # Ending coordinate, here (220, 220) 
    # represents the bottom right corner of rectangle 
    end_point = (row[3], row[4]) 
    # Blue color in BGR 
    color = (255, 0, 0) 
    # Line thickness of 2 px 
    thickness = 2

    image = cv2.rectangle(img, start_point, end_point, color, thickness) 


# Displaying the image  
window_name = 'Image'

cv2.imshow(window_name, image)

# Maintain output window utill 
# user presses a key 

cv2.waitKey(0)         
  
# Destroying present windows on screen 
cv2.destroyAllWindows()  