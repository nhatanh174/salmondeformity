# import cv2

# with open("segment_verterbrae/data_yolo/labels/train/6_1.txt","r") as f:
#     data = f.readlines()
# img = cv2.imread("segment_verterbrae/data_yolo/images/train/6_1.jpg")
# h,w = img.shape[:2]

# for label in data:
#     x_c,y_c,w_label,h_label = label.split(" ")[1:5]
#     w_label = round(float(w_label) * w)
#     h_label = round(float(h_label) * h)
#     x_min = round((float(x_c)*w) - w_label/2)
#     y_min = round((float(y_c)*h) - h_label/2)
#     cv2.rectangle(img,(x_min,y_min),(x_min+w_label,y_min+h_label),(0,0,255),2)
# cv2.imwrite("demo.jpg",img)

a = dict().fromkeys(range(5),[])
print(a)