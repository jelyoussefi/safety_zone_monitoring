import os
import argparse
import fire
import cv2
import numpy as np
import time
import json
import threading
from threading import Thread, Condition
from flask import Flask, redirect, url_for, render_template, make_response, jsonify, request, Response
from flask_bootstrap import Bootstrap
from flask_restful import Resource, Api, reqparse
from utils.yolo_model import YoloV11Model
from utils.images_capture import VideoCapture
import openvino.runtime as ov

class SafetyZoneMonitor:
    def __init__(self, detection_model, input, config_file, device="GPU"):
        with open(config_file) as f:
            self.config = json.load(f)
        self.model = YoloV11Model(detection_model, device, callback_function=None)
        self.input = input
        self.cap = VideoCapture(input)
        self.config_file = config_file
        self.app = Flask(__name__)
        self.port = 80
        self.running = False
        self.cv = Condition()
        self.current_safety_rois = self.config.get('safety_rois', [])
        self.frame = None

    def run(self):
        app = self.app
        Bootstrap(app)

        @app.route('/', methods=['GET', 'POST'])
        def home():
            return render_template('index.html', config=self.config)

        @app.route('/config', methods=['GET', 'POST'])
        def config():
            return render_template('marker.html', config=self.config)

        @app.route('/video_feed')
        def video_feed():
            return Response(self.video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/get_detections', methods=['POST'])
        def get_detections():
            detections = []
            if self.frame is not None:
                resized_image = self.model.preprocess(self.frame)
                self.model.infer_request.set_tensor(self.model.input_layer_ir, ov.Tensor(resized_image))
                self.model.infer_request.infer()
                boxes = self.model.infer_request.results[self.model.compiled_model.output(0)]
                boxes, scores, class_ids = self.model.postprocess(boxes, self.frame)

                safety_rois = self.config.get('safety_rois', [])
                for box, score, class_id in zip(boxes, scores, class_ids):
                    if self.model.labels[class_id] == 'person':
                        x1, y1, x2, y2 = map(int, box)
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        for roi in safety_rois:
                            polygon = np.array(roi['coords'], np.int32).reshape(-1, 1, 2)
                            if cv2.pointPolygonTest(polygon, (center_x, center_y), False) >= 0:
                                detections.append({
                                    'class': 'person',
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': float(score),
                                    'roi_id': roi['id'],
                                    'roi_name': roi['name']
                                })
            return jsonify(detections)

        @app.route('/save', methods=['POST'])
        def save():
            with self.cv:
                data = request.get_json()
                self.current_safety_rois = data.get('safety_rois', self.current_safety_rois)
                self.config['safety_rois'] = self.current_safety_rois
                with open(self.config_file, "w") as f:
                    f.write(json.dumps(self.config, indent=2))
                return jsonify(success=True)

        self.app.run(host='0.0.0.0', port=str(self.port), debug=False, threaded=True)

    def is_in_roi(self, bbox, roi):
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        polygon = np.array(roi, np.int32).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(polygon, (center_x, center_y), False) >= 0

    def video_stream(self):
        self.running = True
        if self.cap is None:
            self.cap = VideoCapture(self.input)

        while self.running:
            with self.cv:
                if self.cap is not None:
                    self.frame = self.cap.read()
                    if self.frame is None:
                        break

                    # Resize frame to 1280x720
                    frame = cv2.resize(self.frame, (1280, 720))
                    frame = frame.copy()
                    for roi in self.current_safety_rois:
                        points = np.array(roi['coords'], np.int32).reshape(-1, 1, 2)
                        cv2.polylines(frame, [points], True, (0, 0, 255), 2)
                        cv2.putText(frame, roi['name'], (int(roi['coords'][0][0]) + 5, int(roi['coords'][0][1]) + 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    frame = self.model.predict(self.frame)
                   
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            time.sleep(0.025)  # Simulate cv2.waitKey(25)

        with self.cv:
            self.cap = None
            self.running = False

def main(detection_model, input, config, device, **kwargs):
    szm = SafetyZoneMonitor(detection_model, input, config, device)
    szm.run()

if __name__ == "__main__":
    fire.Fire(main)