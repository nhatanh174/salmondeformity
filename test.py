import matplotlib
matplotlib.use("TkAgg")
import json
import numpy as np
import glob
from PIL import Image
import os
import cv2 
import matplotlib.pyplot as plt
import io
import glob

# list_img = glob.glob("body/*.jpg")
list_img = ["Report/hinh_anh/data_train_yolo/1_1.jpg","Report/hinh_anh/data_train_yolo/1_2.jpg"]
for path_img in  list_img:
    filename = os.path.basename(path_img).split(".")[0]
    # path_json = "Report/hinh_anh/image/{}.json".format(filename)
    # with open(path_json) as f:
    #     data = json.load(f)
    # points = np.array(data["shapes"][0]["points"],dtype= np.int32)
    img = cv2.imread(path_img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4)
    img = clahe.apply(img)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)    # crop_4_regions = [(0,w//4+50),(w//4-50,w//2+50),(w//2-50,w*3//4+50),(w*3//4-50,w)] 
    # for i,region in enumerate(crop_4_regions):
    #     img_crop = img[:,region[0]:region[1]]
    # mask = np.zeros((w,h)).astype(np.int32)
    # mask = cv2.fillPoly(mask,[points],255)
    # cv2.imwrite("mask_21.jpg",mask)
    # rect_min  = cv2.boundingRect(points)
    # xmin,ymin,w,h = rect_min
    # img_crop = img[ymin:ymin+h,xmin:xmin+w]
    # cv2.drawContours(img,[box],0,(0,0,255),5)
    cv2.imwrite("Report/hinh_anh/anh_CLAHE/{}.jpg".format(filename),img)

# girls_grades = [89, 90, 70, 89, 100, 80, 90, 100, 80, 34]
# boys_grades = [30, 29, 49, 48, 100, 48, 38, 45, 20, 30]
# grades_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# fig=plt.figure()
# ax=fig.add_axes([0,0,1,1])
# ax.scatter(grades_range, girls_grades, color='r')
# ax.scatter(grades_range, boys_grades, color='b')
# ax.set_xlabel('Grades Range')
# ax.set_ylabel('Grades Scored')
# ax.set_title('scatter plot')
# img_buf = io.BytesIO()
# fig.savefig(img_buf, format='png')

# im = Image.open(img_buf)
# im = np.array(im)

# list_imgs = glob.glob("/home/anhnn/Downloads/Pytorch-UNet-master/segment_verterbrae/new_imgs/val/*.jpg")
# list_json = "/home/anhnn/Downloads/Pytorch-UNet-master/segment_verterbrae/new_jsons/"

# for img_path in list_imgs:
#     img= cv2.imread(img_path)
#     h_img,w_img = img.shape[:2]
#     base_name = os.path.basename(img_path)
#     json_path = os.path.join(list_json,base_name.split(".")[0]+".json")
    
#     mask = np.zeros((h_img,w_img),dtype=np.float32)
#     with open(json_path,"r") as f:
#         data = json.load(f)
#     data_shapes = data["shapes"]
#     for label in data_shapes:
#         points = np.array(label["points"],dtype=np.int32)
#         cv2.fillPoly(img=mask,pts=[points],color=1)
#     print(set(mask.flatten()))
    # cv2.imwrite(os.path.join("/home/anhnn/Downloads/Pytorch-UNet-master/segment_verterbrae/new_masks/val",base_name),mask)
# list_mask = glob.glob("/home/anhnn/Downloads/Pytorch-UNet-master/segment_verterbrae/masks/val/*.jpg")
# for mask_test in list_mask:
#     mask = cv2.imread(mask_test)
#     mask_gray = (cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)/255).astype(np.int32)
#     cv2.imwrite(mask_test,mask_gray)