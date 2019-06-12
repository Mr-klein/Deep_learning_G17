# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:26:57 2019

@author: Zhouxin
"""

# =============================================================================
# Gray scale to RGB, or something similar with three channels
# =============================================================================

#https://www.kaggle.com/c/digit-recognizer/discussion/33384


#%% Turn this code into something useful, reverse engineer it for gray to color
# get original image parameters...
width, height = img_file.size
format = img_file.format
mode = img_file.mode

# Make image Greyscale
img_grey = img_file.convert('L')
#img_grey.save('result.png')
#img_grey.show()

# Save Greyscale values
value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
value = value.flatten()
print(value)
with open("img_pixels.csv", 'a') as f:
    writer = csv.writer(f)
    writer.writerow(value)
    
    
    
#   https://discuss.pytorch.org/t/best-way-to-deal-with-1-channel-images/26699/3