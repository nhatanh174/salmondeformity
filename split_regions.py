import cv2
import glob
import numpy as np
import json
import os

list_imgs = glob.glob("/home/anhnn/Downloads/Pytorch-UNet-master/segment_verterbrae/imgs/val/*.jpg")
list_txt = "/home/anhnn/Downloads/Pytorch-UNet-master/segment_verterbrae/body_jsons"
list_json = "/home/anhnn/Downloads/Pytorch-UNet-master/segment_verterbrae/jsons"
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
        points = np.array(shape["points"])
        xshape_min = np.min(points,axis =0)[0].astype(float)
        xshape_max = np.max(points,axis =0)[0].astype(float)
        for index,crop_s in enumerate(crop_4s):
            x_crop_min,x_crop_max = crop_s
            if xshape_min > x_crop_min and xshape_max < x_crop_max:
                new_points = []
                for point in points:
                    new_points.append([point[0]-x_crop_min,point[1] - float(y_crop)])
                shape["points"] = new_points
                s[index+1].append(shape)
                break
    data1["shapes"] = s[1]
    data2["shapes"] = s[2]
    data3["shapes"] = s[3]
    data4["shapes"] = s[4]
    
    with open("segment_verterbrae/new_jsons/"+file_name+"_1.json","w") as f:
        json.dump(data1,f)
    with open("segment_verterbrae/new_jsons/"+file_name+"_2.json","w") as f:
        json.dump(data2,f)
    with open("segment_verterbrae/new_jsons/"+file_name+"_3.json","w") as f:
        json.dump(data3,f)
    with open("segment_verterbrae/new_jsons/"+file_name+"_4.json","w") as f:
        json.dump(data4,f)
    
    for i in range(4):
        x_crop_min,x_crop_max = crop_4s[i] 
        img_crop = img[y_crop:y_crop+h_crop,x_crop_min:x_crop_max]
        cv2.imwrite("segment_verterbrae/new_imgs/val/"+file_name+"_"+str(i+1)+".jpg",img_crop)
        
    