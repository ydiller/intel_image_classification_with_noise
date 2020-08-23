import os
import cv2 as cv
import numpy as np
from tifffile import imsave
from add_noise import add_noise
from PIL import Image

src_dir='./small_data'
dst_dir='./aug_data_small'
for directory in os.listdir(src_dir):
    #for root, dirs, files in os.walk(os.path.join(src_dir,directory)):
    for dir in os.listdir(os.path.join(src_dir,directory)):
        for file in os.listdir(os.path.join(src_dir,directory,dir)):
            path=os.path.join(src_dir,directory,dir,file)
            print(path)
            img = cv.imread(path)
            if(img.shape!=(150,150,3)):
                img=cv.resize(img,(150,150))
            noisy_image = add_noise(img)
            dst_path=os.path.join(dst_dir,directory,dir,file)
            #cv.imwrite(dst_path,noisy_image)
            dst_path2=dst_path[:-4]
            imsave(dst_path2+'.tif', noisy_image)


