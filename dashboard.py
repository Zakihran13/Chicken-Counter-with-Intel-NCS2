from hashlib import new
from PyQt5 import uic
from PyQt5.QtMultimedia import QCameraInfo
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer, QDateTime, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
import cv2,time,sys,sysinfo
import numpy as np
import random as rnd
import pandas as pd
import logging
import os
import shutil
import pathlib
import sys
import cv2
import time
from datetime import datetime
from openvino.inference_engine import IENetwork, IECore

from detect import Inference, YoloParams

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

'''class boardInfoClass(QThread):
    cpu = pyqtSignal(float)
    ram = pyqtSignal(tuple)
    temp = pyqtSignal(float)
    
    def run(self):
        self.ThreadActive = True
        while self.ThreadActive:
            cpu = sysinfo.getCPU()
            ram = sysinfo.getRAM()
            self.cpu.emit(cpu)
            self.ram.emit(ram)

    def stop(self):
        self.ThreadActive = False
        self.quit()'''

class MyGUI(QMainWindow):
    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi('MainWindow.ui', self)

        # atur pilihan sumber video
        self.sumberBox.addItem('Pilih Sumber')
        self.sumberBox.setCurrentText('Pilih Sumber')
        self.sumberBox.currentIndexChanged.connect(self.set_pilihan)

        # set resources info
        #self.resources_usage = boardInfoClass()
        #self.resources_usage.start()
        #self.resources_usage.cpu.connect(self.getCPU_usage)
        #self.resources_usage.ram.connect(self.getRAM_usage)

        # load data
        self.filename = None
        self.count_file = 1
        self.pilihButton.clicked.connect(self.loadData)
        
        # mulai inferensi
        self.inf = Inference()
        self.model = './model/yolov5s.xml'
        self.labels = './model/yolov5s.yaml'
        self.data_scan = list()
        self.mulaiButton.clicked.connect(self.pilihanProsesInferensi)
        self.selesaiButton.clicked.connect(self.jumlahTotal)

        # tombol tutup ditekan
        self.tutupButton_2.clicked.connect(self.tutup)

    def pilih_kamera(self):
        self.kameraBox.setEnabled(True)
        self.pilihButton.setEnabled(False)

        # set list kamera
        self.online_cam = QCameraInfo.availableCameras()
        self.kameraBox.addItems([c.description() for c in self.online_cam])

    def pilih_browser(self):
        self.kameraBox.setEnabled(False)
        self.pilihButton.setEnabled(True)

    def set_pilihan(self, index):
        if str(self.sumberBox.currentText()) == 'Kamera':
            self.pilih_kamera()
        else:
            self.pilih_browser()

    def loadData(self):
        self.filename = QFileDialog.getOpenFileName(filter='Video (*.mp4)')[0]
        self.fileChosen.setText(self.filename.split('/')[-1])
        
    def capture_realtime(self):
        # The duration in seconds of the video captured
        capture_duration = 10

        # kamera terpilih
        cap = cv2.VideoCapture(0)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./temp/history_{}_{}.avi'.format(str(datetime.now().strftime("%Y_%m_%d")), self.count_file), fourcc, 20.0, (640,480))

        start_time = time.time()
        while (int(time.time() - start_time) < capture_duration):
            ret, frame = cap.read()
            if ret==True:
                out.write(frame)

                '''# tampilkan rekaman ke main display
                frame = cv2.resize(frame, (511, 300))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
                self.mainDisplay.setPixmap(QPixmap.fromImage(frame)) ''' 
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        return cv2.VideoCapture('./temp/history_{}_{}.avi'.format(str(datetime.now().strftime("%Y_%m_%d")), self.count_file))
    
    def pilihanProsesInferensi(self):
        if str(self.sumberBox.currentText()) == 'Kamera':
            self.sumberKamera()
        else:
            self.sumberVideo()

    def sumberKamera(self):
        # inferensi
        self.detector(count_file=self.count_file, 
                        model=self.model,
                        input='cam', 
                        labels=self.labels, 
                        prob_threshold=0.28, 
                        iou_threshold=0.3)

        # increment urutan file video yang udah di inferensi
        self.count_file += 1

        # tampilkan riwayat inferece
        self.tampilRiwayat()
        self.jumlahSekarang()

    def sumberVideo(self):
        # infernsi
        self.detector(count_file=self.count_file, 
                        model=self.model,
                        input=self.filename, 
                        labels=self.labels, 
                        prob_threshold=0.28, 
                        iou_threshold=0.3)

        # increment urutan file video yang udah di inferensi
        self.count_file += 1

        # tampilkan riwayat inferece
        self.tampilRiwayat()
        self.jumlahSekarang()
    
    def tampilRiwayat(self):
        data_dir = './data/data inference.csv'
        self.data_all = pd.read_csv(data_dir)

        # read and processing infered image
        hist1 = cv2.imread(self.data_all.frame_infer.values[-2])
        hist1 = cv2.resize(hist1, (self.history1.width(), self.history1.height()))
        hist1 = cv2.cvtColor(hist1, cv2.COLOR_BGR2RGB)
        self.history_img1 = QImage(hist1, hist1.shape[1], hist1.shape[0], hist1.strides[0], QImage.Format_RGB888)
        self.history1.setPixmap(QPixmap.fromImage(self.history_img1))

        hist2 = cv2.imread(self.data_all.frame_infer.values[-3])
        hist2 = cv2.resize(hist2, (self.history2.width(), self.history2.height()))
        hist2 = cv2.cvtColor(hist2, cv2.COLOR_BGR2RGB)
        self.history_img2 = QImage(hist2, hist2.shape[1], hist2.shape[0], hist2.strides[0], QImage.Format_RGB888)
        self.history2.setPixmap(QPixmap.fromImage(self.history_img2))
    
    def jumlahSekarang(self):
        data_all = pd.read_csv('./data/data inference.csv')
        current_obj = data_all.jumlah_obj.values[-1].copy()
        self.sekarangLcd.display(current_obj)

    def jumlahTotal(self):
        total = sum(self.data_scan)
        self.totalLcd.display(total)
        self.data_scan.clear()

    def tutup(self):
        #self.resources_usage.stop()
        self.count_file = 1
        sys.exit(app.exec_())


    # ===================== INFERENCE FUNCTION ==================================
    def detector(self, 
                count_file = 0,
                model = None,
                input = None,
                cpu_extention = None,
                device = 'CPU',
                labels = None,
                prob_threshold = 0.5,
                iou_threshold = 0.4,
                number_iter = 1,
                perf_count = False,
                raw_output_message = False,
                no_show = False,
                save_img = True):
        #args = build_argparser().parse_args()
        count_file = count_file
        model = model
        input = input
        cpu_extention = cpu_extention
        device = device
        labels = labels
        prob_threshold = prob_threshold
        iou_threshold = iou_threshold
        number_iter = number_iter
        perf_count = perf_count
        raw_output_message = raw_output_message
        no_show = no_show
        save_img = save_img

        # ------------- 1. Plugin initialization for specified device and load extensions library if specified -------------
        log.info("Creating Inference Engine...")
        ie = IECore()
        if cpu_extention and 'CPU' in device:
            ie.add_extension(cpu_extention, "CPU")

        # -------------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) --------------------
        model = model
        log.info(f"Loading network:\n\t{model}")
        net = ie.read_network(model=model)

        # ---------------------------------- 3. Load CPU extension for support specific layer ------------------------------

        assert len(net.input_info.keys()) == 1, "Sample supports only YOLO V3 based single input topologies"

        # ---------------------------------------------- 4. Preparing inputs -----------------------------------------------
        log.info("Preparing inputs")
        input_blob = next(iter(net.input_info))

        #  Defaulf batch_size is 1
        net.batch_size = 1

        # Read and pre-process input images
        n, c, h, w = net.input_info[input_blob].input_data.shape

        if labels:
            with open(labels, 'r') as f:
                labels_map = [x.strip() for x in f]
        else:
            labels_map = None


        # read video file
        if input == "cam":
            cap = self.capture_realtime()
        else:
            cap = self.inf.capture_recorded(input, count_file=self.count_file)

        is_async_mode = True
        number_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames

        wait_key_code = 1

        # Number of frames in picture is 1 and this will be read in cycle. Sync mode is default value for this case
        if number_input_frames != 1:
            ret, frame = cap.read()
        else:
            is_async_mode = False
            wait_key_code = 0

        # ----------------------------------------- 5. Loading model to the plugin -----------------------------------------
        log.info("Loading model to the plugin")

        if 'MYRIAD' in ie.available_devices:
            exec_net = ie.load_network(network=net, num_requests=2, device_name='MYRIAD')
        else:
            exec_net = ie.load_network(network=net, num_requests=2, device_name=device)

        cur_request_id = 0
        next_request_id = 1

        # ----------------------------------------------- 6. Doing inference -----------------------------------------------
        log.info("Starting inference...")
        print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
        print("To switch between sync/async modes, press TAB key in the output window")

        # list sementara
        original_frame = []
        infered_frame = []
        jml_obj = []

        # checking data file existance
        if os.path.exists('./data/data inference.csv'):
            data_all = pd.read_csv('./data/data inference.csv')
        else:
            data_all = pd.DataFrame(columns=['tgl', 'frame_ori', 'frame_infer', 'jumlah_obj'])


        while cap.isOpened():
            # Here is the first asynchronous point: in the Async mode, we capture frame to populate the NEXT infer request
            # in the regular mode, we capture frame to the CURRENT infer request
            if is_async_mode:
                ret, next_frame = cap.read()
            else:
                ret, frame = cap.read()

            if not ret:
                break

            if is_async_mode:
                request_id = next_request_id
                in_frame = self.inf.letterbox(frame, (w, h))
            else:
                request_id = cur_request_id
                in_frame = self.inf.letterbox(frame, (w, h))

            # resize input_frame to network size
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))

            # Start inference
            exec_net.start_async(request_id=request_id, inputs={input_blob: in_frame})

            # save ori image
            original_frame.append(frame)

            # Collecting object detection results
            objects = list()
            if exec_net.requests[cur_request_id].wait(-1) == 0:
                output = exec_net.requests[cur_request_id].output_blobs
                for layer_name, out_blob in output.items():
                    layer_params = YoloParams(side=out_blob.buffer.shape[2])
                    log.info("Layer {} parameters: ".format(layer_name))
                    layer_params.log_params()
                    objects += self.inf.parse_yolo_region(out_blob.buffer, in_frame.shape[2:],
                                                #in_frame.shape[2:], layer_params,
                                                frame.shape[:-1], layer_params,
                                                prob_threshold)

            # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
            objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
            for i in range(len(objects)):
                if objects[i]['confidence'] == 0:
                    continue
                for j in range(i + 1, len(objects)):
                    if self.inf.intersection_over_union(objects[i], objects[j]) > iou_threshold:
                        objects[j]['confidence'] = 0

            # Drawing objects with respect to the --prob_threshold CLI parameter
            objects = [obj for obj in objects if obj['confidence'] >= prob_threshold]

            if len(objects) and raw_output_message:
                log.info("\nDetected boxes for batch {}:".format(1))
                log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX | COLOR ")

            # bounding-box plotting
            origin_im_size = frame.shape[:-1]
            print(origin_im_size)
            total_object = len(objects)
            for i, obj in enumerate(objects):
                # Validation bbox of detected object
                if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
                    continue
                color = (int(min(obj['class_id'] * 12.5, 255)),
                        min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
                det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
                    str(obj['class_id'])

                if raw_output_message:
                    log.info(
                        "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} | {} ".format(det_label, obj['confidence'], obj['xmin'],
                                                                                obj['ymin'], obj['xmax'], obj['ymax'],
                                                                                color))

                frame = cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
                frame = cv2.putText(frame,
                                    "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                                    (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
                frame = cv2.putText(frame, str(total_object), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 4, (255,255,255), 3)

            # save infered img
            infered_frame.append(frame)
            jml_obj.append(total_object)

            if not no_show:
                #cv2.imshow("DetectionResults", frame)  
                inframe = cv2.resize(frame, (self.mainDisplay.width(), self.mainDisplay.height()))
                inframe = cv2.cvtColor(inframe, cv2.COLOR_BGR2RGB)
                inframe = QImage(inframe, inframe.shape[1], inframe.shape[0], inframe.strides[0], QImage.Format_RGB888)
                self.mainDisplay.setPixmap(QPixmap.fromImage(inframe))          
            
            if is_async_mode:
                cur_request_id, next_request_id = next_request_id, cur_request_id
                frame = next_frame

            if not no_show:
                key = cv2.waitKey(wait_key_code)
        
                # ESC key
                if key == 27:
                    break
                # Tab key
                if key == 9:
                    exec_net.requests[cur_request_id].wait()
                    is_async_mode = not is_async_mode
                    log.info("Switched to {} mode".format("async" if is_async_mode else "sync"))
                

        # save data to dataframe
        id_max = jml_obj.index(np.max(jml_obj))

        # writing image and add to dataframe
        original_path = './img/original/'
        inference_path = './img/inference/'

        name_original = original_path + 'original_{}_{}'.format(str(datetime.now().strftime("%Y_%m_%d")), count_file) + '.jpg'
        name_inference = inference_path + 'inference_{}_{}'.format(str(datetime.now().strftime("%Y_%m_%d")), count_file) + '.jpg'

        cv2.imwrite(name_original, original_frame[id_max])
        cv2.imwrite(name_inference, infered_frame[id_max])

        # add new values to dataframe
        new_val = {'tgl':str(datetime.now().strftime("%Y-%m-%d")), 'frame_ori':name_original, 'frame_infer':name_inference, 'jumlah_obj':int(jml_obj[id_max])}
        data_all = data_all.append([new_val], ignore_index=False)

        # simpan data scan sementara
        self.data_scan.append(int(jml_obj[id_max]))

        # export dataframe
        data_all.to_csv('./data/data inference.csv', index=False)

        # delete all infered value
        original_frame.clear()
        infered_frame.clear()
        jml_obj.clear()

        # destroy all unused window
        cv2.destroyAllWindows()
        

if __name__=='__main__':
    app = QApplication([])
    window = MyGUI()
    window.show()
    app.exec_()