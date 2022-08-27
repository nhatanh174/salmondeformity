from ntpath import join
import imgaug.augmenters as iaa
import cv2
import os 
import random
import glob
import numpy as np
import shutil

img_folder_path = "/hdd/anhnn/Pytorch-UNet-master/total_data_yolo/data_old/images/val"
# mask_folder_path = "/home/anhnn/Downloads/Pytorch-UNet-master/segment_verterbrae/masks/train"
result_folder_path = "/hdd/anhnn/Pytorch-UNet-master/total_data_yolo/data_old/images/val"
list_file = glob.glob(os.path.join(img_folder_path,"*.jpg"))
for file in list_file:
    file_name = os.path.basename(file).split(".")[0]
    img = cv2.imread(os.path.join(img_folder_path,file))
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4)
    img_aug = clahe.apply(img_gray)
    img_aug = cv2.cvtColor(img_aug,cv2.COLOR_GRAY2BGR)
    # mask = cv2.imread(os.path.join(mask_folder_path,file))

    # angle = random.randint(-10,10)
    # augment = iaa.LinearContrast(4)
    # img_aug = augment.augment_image(img)
    # mask_aug = augment.augment_image(mask)
    # mask_aug = np.array(mask_aug/255,dtype = np.int8)

    cv2.imwrite(os.path.join(result_folder_path,file_name+".jpg"),img_aug)
    # shutil.copyfile(os.path.join(mask_folder_path,file_name+".jpg"),os.path.join(mask_folder_path,file_name+"_x.jpg"))
    # cv2.imwrite(os.path.join(mask_folder_path,file_name+"_x.jpg"),mask_aug)
