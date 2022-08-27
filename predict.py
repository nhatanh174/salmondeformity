import argparse
from cmath import sqrt
import json
import logging
import os
from tabnanny import check
import cv2
import glob
from tqdm import tqdm
import imgaug.augmenters as iaa
from shapely.geometry import Polygon
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utilss.data_loading import BasicDataset
from unet import UNet
from utilss.utils import plot_img_and_mask
from time import perf_counter, time
import pickle
import math
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import sys
import io

sys.path.insert(0,"./yolov5")
from yolov5.infer import get_predict,attempt_load

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def find_IOU(box1,box2):
    x_c1,y_c1,w_c1,h_c1,conf1 = box1
    x_c2,y_c2,w_c2,h_c2,conf2 = box2

    x_min1 = x_c1 - w_c1//2
    y_min1 = y_c1 - h_c1//2
    x_max1 = x_c1 + w_c1//2
    y_max1 = y_c1 + h_c1//2

    x_min2 = x_c2 - w_c2//2
    y_min2 = y_c2 - h_c2//2
    x_max2 = x_c2 + w_c2//2
    y_max2 = y_c2 + h_c2//2

    x_a = max(x_min1,x_min2)
    y_a = max(y_min1,y_min2)
    x_b = min(x_max1,x_max2)
    y_b = min(y_max1,y_max2)

    inter_area = max(0,x_b - x_a) * max(0,y_b -y_a)
    bbox1_area = w_c1 * h_c1
    bbox2_area = w_c2 * h_c2
    
    iou = inter_area / (bbox1_area + bbox2_area - inter_area)
    return iou
    
def check_aaa(new_overlaps,ver_append):
    kq = True
    if len(new_overlaps) == 0:
        return kq
    for ver in new_overlaps:
        if find_IOU(ver,ver_append) > 0.7:
            kq = False
            break
    return kq

def remove_duplicate_object(vers,overlap_regions):
    vers_cp = []
    ver_overlaps_all = dict()
    for ver in vers:
        x_c,y_c,w,h,conf = ver 
        check = 0
        for index,overlap_region in enumerate(overlap_regions):
            x_min,x_max  = overlap_region
            if x_c - w/2 > x_min and x_c + w/2 <x_max:
                if index not in list(ver_overlaps_all.keys()):
                    ver_overlaps_all[index] = list()
                    ver_overlaps_all[index].append(ver)
                else:
                    ver_overlaps_all[index].append(ver)
                check = 1
                break
        if check == 0:
            vers_cp.append(ver)
    for ind in list(ver_overlaps_all.keys()):
        ver_overlaps = ver_overlaps_all[ind]
        new_overlaps = []
        for i in range(len(ver_overlaps)):
            ver_1 = ver_overlaps[i]
            for j in range(i,len(ver_overlaps)):
                ver_2 = ver_overlaps[j]
                if find_IOU(ver_2,ver_1) > 0.7:
                    ver_append = ver_2 if ver_2[-1] > ver_1[-1] else ver_1
                    if ver_append not in new_overlaps and check_aaa(new_overlaps,ver_append) == True:
                        new_overlaps.append(ver_append)
        vers_cp.extend(new_overlaps)
    return vers_cp

def get_new_box(box1,box2):
    xmin1,ymin1,xmax1,ymax1 = box1
    xmin2,ymin2,xmax2,ymax2 = box2
    return [min(xmin1,xmin2),min(ymin1,ymin2),max(xmax1,xmax2),max(ymax1,ymax2)]

def remove_duplicate_vertebrae(vers,overlap_regions,h):
    for index in range(len(overlap_regions)):
        xmin,xmax = overlap_regions[index]
        polygon_overlap = Polygon([(xmin,0),(xmin,h),(xmax,h),(xmax,0)]).buffer(0)

        ver_in_region1 = vers[index]
        ver_in_region2 = vers[index+1]

        new_ver_1 = []
        new_ver_2 = []

        ver_in_intersection1 = []
        polygon_in_intersection1 = []
        overlap_in_intersection1 = []

        ver_in_intersection2 = []
        polygon_in_intersection2 = []
        overlap_in_intersection2 = []

        for ver in ver_in_region1:
            xmin,ymin,xmax,ymax = ver
            polygon_ver = Polygon([(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin)]).buffer(0)
            if not polygon_ver.intersects(polygon_overlap):
                new_ver_1.append(ver)
            else:
                ver_in_intersection1.append(ver)
                polygon_in_intersection1.append(polygon_ver)
                overlap_in_intersection1.append(polygon_ver.intersection(polygon_overlap))
        
        for ver in ver_in_region2:
            xmin,ymin,xmax,ymax = ver
            polygon_ver = Polygon([(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin)]).buffer(0)
            if not polygon_ver.intersects(polygon_overlap):
                new_ver_2.append(ver)
            else:
                ver_in_intersection2.append(ver)
                polygon_in_intersection2.append(polygon_ver)
                overlap_in_intersection2.append(polygon_ver.intersection(polygon_overlap))
        label2_index = []
        for i in range(len(ver_in_intersection1)):
            ver1 = ver_in_intersection1[i]
            ver1_polygon = polygon_in_intersection1[i]
            ver1_overlap = overlap_in_intersection1[i]
            duplicate_flag = False
            for j in range(len(ver_in_intersection2)):
                ver2 = ver_in_intersection2[j]
                ver2_polygon = polygon_in_intersection2[j]
                ver2_overlap = overlap_in_intersection2[j]

                if ver1_overlap.intersects(ver2_overlap):
                    inter = ver1_overlap.intersection(ver2_overlap)
                    iou = inter.area/(ver1_overlap.area+ver2_overlap.area - inter.area)
                    if iou > 0.7 or (ver1_overlap.area != 0 and inter.area/ver1_overlap.area > 0.9) or (ver2_overlap.area !=0 and inter.area/ver2_overlap.area > 0.9):
                        duplicate_flag = True
                        new_ver_1.append(get_new_box(ver1,ver2))
                        label2_index.append(j)
            if not duplicate_flag:
                new_ver_1.append(ver1)
        for j in range(len(ver_in_intersection2)):
            if j not in label2_index:
                new_ver_2.append(ver_in_intersection2[j])
        vers[index] = new_ver_1
        vers[index+1] = new_ver_2

    
    kq = []
    for item in list(vers.values()):
        kq += item
    return kq
    

def sort(vers,reserve= True):
    def my_func(e):
        return e[0]
    vers.sort(reverse= reserve,key = my_func)
    return vers
    
def distance(current_ver,after_ver):
    current_center = ((current_ver[0]+current_ver[2])/2,(current_ver[1]+current_ver[3])/2)
    after_center = ((after_ver[0]+after_ver[2])/2,(after_ver[1]+after_ver[3])/2)
    vector_kc = np.array([after_center[0]-current_center[0],after_center[1]-current_center[1]])
    kc = np.linalg.norm(vector_kc)
    return kc

def check_overlap(box1,box2):
    x_c1,y_c1,w_c1,h_c1,conf1 = box1
    x_c2,y_c2,w_c2,h_c2,conf2 = box2

    x_min1 = x_c1 - w_c1//2
    y_min1 = y_c1 - h_c1//2
    x_max1 = x_c1 + w_c1//2
    y_max1 = y_c1 + h_c1//2

    x_min2 = x_c2 - w_c2//2
    y_min2 = y_c2 - h_c2//2
    x_max2 = x_c2 + w_c2//2
    y_max2 = y_c2 + h_c2//2

    x_a = max(x_min1,x_min2)
    y_a = max(y_min1,y_min2)
    x_b = min(x_max1,x_max2)
    y_b = min(y_max1,y_max2)

    inter_area = max(0,x_b - x_a) * max(0,y_b -y_a)
    if inter_area > 0:
        return True
    else:
        return False

def distance_point2line(point,start_point,end_point):
    k = (end_point[1]-start_point[1])/(end_point[0]-start_point[0])
    y = int((k**2*point[1]  + point[0]*k - k*start_point[0] +start_point[1])/(1+k**2))
    x = int(k*(point[1]-y) + point[0])
    dis = sqrt((point[0]-x)**2 + (point[1]-y)**2)
    return round(dis.real)


def line_targent(x,point_x,point_y,derivate):
    return derivate*(x-point_x) + point_y

def draw_cobb_angle(center_line_x,center_line_y,point_max,point_min,T_max,T_min,limit_plot,index):
    point_max_x,point_max_y = point_max
    point_min_x,point_min_y = point_min
    plt.plot(center_line_x,center_line_y)
    plt.scatter(center_line_x,center_line_y,s=10)
    plt.scatter(point_max_x,point_max_y,color ="red")
    plt.scatter(point_min_x,point_min_y,color ="red")

    xrangemax = np.linspace(point_max_x-100, point_max_x+100, 200)
    yrangemax = []
    for item in xrangemax:
        yrangemax.append(line_targent(item, point_max_x, point_max_y,T_max))
    plt.plot(xrangemax, yrangemax,'C1--', linewidth = 2)

    xrangemin = np.linspace(point_min_x-100, point_min_x+100, 200)
    yrangemin = []
    for item in xrangemin:
        yrangemin.append(line_targent(item, point_min_x, point_min_y,T_min))
    plt.plot(xrangemin, yrangemin,'C1--', linewidth = 2)
    plt.ylim([limit_plot[1][0],limit_plot[1][1]])
    plt.xlim([limit_plot[0][0],limit_plot[0][1]])
    plt.xlabel("width")
    plt.ylabel("height")
    plt.gca().invert_yaxis()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.savefig("{}.jpg".format(str(index)))
    plt.close()
    im = Image.open(img_buf)
    im = np.array(im)
    return im

def check_angle(vers,cubic,limit_plot,index):
    center_line = []
    for ver in vers:
        xc = (ver[0]+ver[2])//2
        yc = (ver[1]+ver[3])//2
        center_line.append((xc,yc))
    center_line.sort(key = lambda x: x[0])
    center_line_x =[]
    center_line_y = []
    for point in center_line:
        center_line_x.append(point[0])
        center_line_y.append(point[1])
    pp  = cubic.derivative()

    # find point 
    slopes = []
    for i in range(0,len(center_line_x)):
        slopes.append(pp(center_line_x[i]))
    
    # point max
    index_max = np.argmax(np.array(slopes))
    point_max_x = center_line_x[index_max]
    point_max_y = center_line_y[index_max]

    #point min
    index_min = np.argmin(np.array(slopes))
    point_min_x = center_line_x[index_min]
    point_min_y = center_line_y[index_min]
    
    # T_max = pp(point_max_x)
    T_max = slopes[index_max]
    T_min = slopes[index_min]

    angle = 180/3.14 *abs(math.atan((T_max-T_min )/(1 + T_max*T_min)))
    plot_cobb_angle = draw_cobb_angle(center_line_x,center_line_y,(point_max_x,point_max_y),(point_min_x,point_min_y),T_max,T_min,limit_plot,index)
    return angle,plot_cobb_angle


def inference_LKS(model_segment,model_yolo,device,half,input_img):
    h_imgs,w_imgs = input_img.shape[:2]
    input_img_pillow = Image.fromarray(input_img)
    mask = predict_img(net=model_segment,
                    full_img=input_img_pillow,
                    scale_factor=0.2,
                    out_threshold=0.5,
                    device=device)
    
    # tìm contour có diện tích lớn nhất
    mask = (np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=cv2.contourArea)
    
    # crop max contour
    x,y,w,h = cv2.boundingRect(max_contour)
    body_crop = input_img[y:y+h,x:x+w]
    body_crop_kq =body_crop.copy()

    # preprocessing : change contrast 
    body_crop = cv2.cvtColor(body_crop,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4)
    body_crop = clahe.apply(body_crop)
    body_crop = cv2.cvtColor(body_crop,cv2.COLOR_GRAY2BGR)
    
    # split 4 regions and detect vertebra
    crop_4_regions = [(0,w//4+50),(w//4-50,w//2+50),(w//2-50,w*3//4+50),(w*3//4-50,w)] 
    overlap_regions = [(w//4-50,w//4+50),(w//2-50,w//2+50),(w*3//4-50,w*3//4+50)]
    
    vers =dict()
    for i,region in enumerate(crop_4_regions):
        x_min,x_max = region
        crop_region = body_crop[0:h,x_min:x_max]
        ver_in_region = []
        time1  = perf_counter()
        pred_vertebra = get_predict(crop_region,model_yolo,device,half)
        time2 = perf_counter()
        # print("Time predict 1 image: ",time2 -time1)
        time2 = perf_counter()
        for det in pred_vertebra:
            xmin,ymin,xmax,ymax,conf,cls = det
            xmin = xmin + x_min
            xmax = xmax + x_min
            ver_in_region.append([xmin,ymin,xmax,ymax])
        vers[i] = ver_in_region

    # remove duplicate vertebral of body
    vers_auth  = remove_duplicate_vertebrae(vers,overlap_regions,h)
    
    # align vertebra
    vers_auth = sort(vers_auth,True)
    regions = [vers_auth[0:8],vers_auth[8:31],vers_auth[31:50],vers_auth[50:]]
    kq_regions = []
    limit_plot =[]
    for region in regions:
        box_max = region[0]
        box_min = region[-1]
        limit_plot.append([(int(box_min[0])-5,int(box_max[2])+5),(0,h)])
        kq_regions.append(body_crop_kq[:,int(box_min[0])-5:int(box_max[2])+5])
    
    # find cubic spline
    vers_cubic = []
    for index,region in  enumerate(regions):
        if index in [0]:
            for i in range(0,len(region),2):
                vers_cubic.append(region[i])
        else:
            for i in range(0,len(region),3):
                vers_cubic.append(region[i])

    vers_cubic = sort(vers_cubic,False)
    
    center_x = []
    center_y = []
    for i in range(len(vers_cubic)):
        xmin,ymin,xmax,ymax = vers_cubic[i]
        center_x.append(int((xmin+xmax)/2))
        center_y.append(int((ymin+ymax)/2))
    cubic = CubicSpline(center_x,center_y,bc_type='natural')
    
    # compute cobb with cubic spline
    cobbs = []
    plot_cobbs=[]
    for index,region in enumerate(regions):
        cobb_angle,plot_cobb = check_angle(region,cubic,limit_plot[index],index)
        cobbs.append(cobb_angle)
        plot_cobbs.append(plot_cobb)
    
    # draw box of vertebral
    for i,ver in enumerate(vers_auth):
        xmin_ob,ymin_ob,xmax_ob,ymax_ob = ver
        cv2.rectangle(body_crop_kq, (int(xmin_ob),int(ymin_ob)),(int(xmax_ob),int(ymax_ob)),(0,0,255),1)
        cv2.putText(body_crop_kq,str(i+1),(int(xmin_ob),int(ymin_ob)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
    
    return body_crop_kq,kq_regions,cobbs,plot_cobbs

def inference_Compress(model_segment,model_yolo,device,half,input_img):
    h_imgs,w_imgs = input_img.shape[:2]
    input_img_pillow = Image.fromarray(input_img)
    mask = predict_img(net=model_segment,
                    full_img=input_img_pillow,
                    scale_factor=0.2,
                    out_threshold=0.5,
                    device=device)
    
    # tìm contour có diện tích lớn nhất
    mask = (np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=cv2.contourArea)
    
    # crop max contour
    x,y,w,h = cv2.boundingRect(max_contour)
    body_crop = input_img[y:y+h,x:x+w]
    body_crop_kq =body_crop.copy()

    # preprocessing : change contrast 
    body_crop = cv2.cvtColor(body_crop,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4)
    body_crop = clahe.apply(body_crop)
    body_crop = cv2.cvtColor(body_crop,cv2.COLOR_GRAY2BGR)
    
    # split 4 regions and detect vertebra
    crop_4_regions = [(0,w//4+50),(w//4-50,w//2+50),(w//2-50,w*3//4+50),(w*3//4-50,w)] 
    overlap_regions = [(w//4-50,w//4+50),(w//2-50,w//2+50),(w*3//4-50,w*3//4+50)]
    
    vers =dict()
    for i,region in enumerate(crop_4_regions):
        x_min,x_max = region
        crop_region = body_crop[0:h,x_min:x_max]
        ver_in_region = []
        time1  = perf_counter()
        pred_vertebra = get_predict(crop_region,model_yolo,device,half)
        time2 = perf_counter()
        # print("Time predict 1 image: ",time2 -time1)
        time2 = perf_counter()
        for det in pred_vertebra:
            xmin,ymin,xmax,ymax,conf,cls = det
            xmin = xmin + x_min
            xmax = xmax + x_min
            ver_in_region.append([xmin,ymin,xmax,ymax])
        vers[i] = ver_in_region

    # remove duplicate vertebral of body
    vers_auth  = remove_duplicate_vertebrae(vers,overlap_regions,h)

    # draw the distance chart of vertebrals 
    vers_auth = sort(vers_auth,True)

    kc =[]
    for i in range(len(vers_auth)-1):
        current_ver = vers_auth[i]
        after_ver = vers_auth[i+1]
        kc.append(distance(current_ver,after_ver))

    # draw box of vertebral
    for i,ver in enumerate(vers_auth):
        xmin_ob,ymin_ob,xmax_ob,ymax_ob = ver
        cv2.rectangle(body_crop_kq, (int(xmin_ob),int(ymin_ob)),(int(xmax_ob),int(ymax_ob)),(0,0,255),1)
        cv2.putText(body_crop_kq,str(i+1),(int(xmin_ob),int(ymin_ob)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
    cv2.imwrite("report_1408/Compress_prediction/{}_predict.jpg".format(file_name.split(".")[0]),body_crop_kq)
    
    plt.xlabel("index of vertebra")
    plt.ylabel("distance (pixel)")
    plt.bar(range(len(vers_auth)-1),kc)
    plt.savefig("report_1408/Compress_prediction/{}.jpg".format(file_name.split(".")[0]))
    plt.close()



if __name__ == '__main__':
    model_yolo_path = "checkpoints/YOLOv5/best.pt"
    model_segmet_path ="checkpoints/UNET/checkpoint_epoch27.pth"

    # img_input = cv2.imread("Report/hinh_anh/image/21.jpg") 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
    # half = device.type != 'cpu'
    half = False
    # load detect model
    model = attempt_load(model_yolo_path, map_location=device)
    if half:
        model.half()
    
    net = UNet(n_channels=3, n_classes=2)
    # VR = VertebraeRecognition(model_path= model_yolov5_path,device=device,half=half,imgsz=(1280,1280),conf_thres=0.7,iou_thres=0.8)
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model_segmet_path, map_location=device))
    
    path_to_folder = "report_1408/Compress/"
    for img_path in glob.glob(os.path.join(path_to_folder,"*.jpg")):
        try:
            img_input = cv2.imread(img_path)
            file_name = os.path.basename(img_path)
            # rotate_img
            json_path = os.path.join(path_to_folder,"{}.json".format(file_name.split(".")[0]))
            with open(json_path) as f:
                data = json.load(f)
            polys = data["shapes"][0]["points"]
            polys = np.array(polys,dtype =np.int32)
            rect_box_min = cv2.minAreaRect(polys)
            box_points = cv2.boxPoints(rect_box_min)
            bbox = np.int0(box_points)
            A,B,C,D = bbox
            AB = np.array([B[0]-A[0],B[1]-A[0]])
            AD = np.array([D[0]-A[0],D[1]-A[1]]) 
            vector_centerline = AB if np.linalg.norm(AB) > np.linalg.norm(AD) else AD
            hesogoc = -1*np.arctan(vector_centerline[1]/vector_centerline[0])*180/3.14
            augmentor = iaa.Rotate(hesogoc,fit_output=True)
            rotated_img = augmentor.augment_image(img_input)
            # cv2.imwrite("report_1408/lks_prediction/{}".format(file_name),rotated_img)
            inference_Compress(net,model,device,half,rotated_img)
        except:
            continue




