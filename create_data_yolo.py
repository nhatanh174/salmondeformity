
import cv2
import glob
import numpy as np
import json
import os

list_imgs = glob.glob("/home/anhnn/Downloads/Pytorch-UNet-master/segment_verterbrae/imgs/val/*.jpg")
list_txt = "/home/anhnn/Downloads/Pytorch-UNet-master/segment_verterbrae/body_jsons"
list_json = "/home/anhnn/Downloads/Pytorch-UNet-master/segment_verterbrae/jsons"

image_yolo = "/home/anhnn/Downloads/Pytorch-UNet-master/segment_verterbrae/data_yolo/images/val/"
label_yolo = "/home/anhnn/Downloads/Pytorch-UNet-master/segment_verterbrae/data_yolo/labels/val/"
for img_path in list_imgs:
    img= cv2.imread(img_path)
    h_img,w_img = img.shape[:2]
    file_name = os.path.basename(img_path).split(".")[0]
    txt_path = os.path.join(list_txt,file_name+ ".txt")
    json_path = os.path.join(list_json,file_name+ ".json")
    
    with open(txt_path,"r") as f:
        regions = f.readline()
    
    x_crop,y_crop,w_crop,h_crop = [ int(item) for item in regions.split(" ")]
    
    crop_4s = [(x_crop,x_crop+w_crop//4),(x_crop + w_crop//4, x_crop+w_crop//2),(x_crop+w_crop//2,x_crop+w_crop*3//4),(x_crop+w_crop*3//4,x_crop+w_crop)] 
    s = {1:[],2:[],3:[],4:[]}
    
    with open(json_path,"r") as f:
        data = json.load(f)
    data_shapes = data["shapes"]
    
    data1 = data.copy()
    data2 = data.copy()
    data3 = data.copy()
    data4 = data.copy()

    for shape in data_shapes:
        points = np.array(shape["points"],dtype= np.float32)
        
        x_yl_min, y_yl_min , w_yl, h_yl = cv2.boundingRect(points)

        for index,crop_s in enumerate(crop_4s):
            x_crop_min,x_crop_max = crop_s
            if x_yl_min > x_crop_min and x_yl_min + w_yl < x_crop_max:
                x_yl_label = ((x_yl_min - x_crop_min)+w_yl/2)/(x_crop_max - x_crop_min )
                y_yl_label = ((y_yl_min- y_crop)+h_yl/2)/h_crop
                w_yl_label = w_yl/(x_crop_max -x_crop_min)
                h_yl_label = h_yl/h_crop
                s[index+1].append("0 {} {} {} {}\n".format(x_yl_label,y_yl_label,w_yl_label,h_yl_label))
                break
       
    for i in range(4):
        x_crop_min,x_crop_max = crop_4s[i] 
        img_crop = img[y_crop:y_crop+h_crop,x_crop_min:x_crop_max]
        cv2.imwrite(image_yolo+file_name+"_"+str(i+1)+".jpg",img_crop)
        with open(label_yolo+file_name+"_"+str(i+1)+ ".txt","w") as f:
            f.writelines(s[i+1])

        
        