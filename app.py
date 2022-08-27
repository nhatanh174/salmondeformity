import base64
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from flask import Flask,render_template,request
from flask_socketio import SocketIO
import json
import torch
import logging
from unet import UNet
from predict import inference
import cv2
import sys
import time
sys.path.insert(0,"./yolov5")
from yolov5.infer import attempt_load

app = Flask(__name__)
# socketio = SocketIO(app)

@app.route("/")
def main():
    return render_template("index.html",data_json =None)

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files['file']
    filename = f.filename
    img = cv2.imread(os.path.join("OC171D",filename))
    time1 = time.perf_counter()
    body_detect,kq_regions,cobbs, plot_cobbs = inference(net,model,device,half,img)
    time2 = time.perf_counter()
    print("Processing {} , time {}".format(filename,time2-time1))

    _,buffer_img = cv2.imencode('.jpg',img)
    img_base64 = base64.b64encode(buffer_img).decode()

    _,buffer_img_body = cv2.imencode('.jpg',body_detect)
    img_body_base64 = base64.b64encode(buffer_img_body).decode()

    regions=[]
    for crop_region in kq_regions:
        _,buffer_img_crop = cv2.imencode('.jpg',crop_region)
        img_crop_base64 = base64.b64encode(buffer_img_crop).decode()
        regions.append(img_crop_base64)
    
    plot_cobbs_b64 =[]
    for plot_cobb in plot_cobbs:
        if plot_cobb is not None:
            _,buffer_plot_cobb = cv2.imencode('.jpg',plot_cobb)
            plot_cobb_base64 = base64.b64encode(buffer_plot_cobb).decode()
            plot_cobbs_b64.append(plot_cobb_base64)
        else:
            plot_cobbs_b64.append(None)

    # socketio.emit("ketqua",{"original":img_base64,"body":img_body_base64,"regions":regions})
    return render_template("index.html",data_json ={"original":img_base64,"body":img_body_base64,"regions":regions , "cobbs":cobbs , "plot_cobbs":plot_cobbs_b64})

if __name__=="__main__":
    model_yolo_path = "weights/best.pt"
    model_segmet_path ="04_10\checkpoint_epoch27.pth"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
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
    app.run(host='0.0.0.0', port=1704, debug=True,use_reloader = False)
