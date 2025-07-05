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
        self.model = YoloV11Model(detection_model, device)
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
    
    def bbox_intersects_roi(self, bbox, roi):
        """Check if bounding box intersects with ROI polygon"""
        x1, y1, x2, y2 = bbox
        
        # Create polygon from ROI coordinates
        polygon = np.array(roi, np.int32).reshape(-1, 1, 2)
        
        # Check if any corner of the bbox is inside the polygon
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for corner in corners:
            if cv2.pointPolygonTest(polygon, corner, False) >= 0:
                return True
        
        # Check if any edge of the bbox intersects with polygon edges
        bbox_edges = [
            ((x1, y1), (x2, y1)),  # top edge
            ((x2, y1), (x2, y2)),  # right edge
            ((x2, y2), (x1, y2)),  # bottom edge
            ((x1, y2), (x1, y1))   # left edge
        ]
        
        roi_points = roi
        for i in range(len(roi_points)):
            roi_edge = (roi_points[i], roi_points[(i + 1) % len(roi_points)])
            for bbox_edge in bbox_edges:
                if self.line_intersects(bbox_edge, roi_edge):
                    return True
        
        # Check if polygon is completely inside bbox
        for point in roi_points:
            if not (x1 <= point[0] <= x2 and y1 <= point[1] <= y2):
                break
        else:
            # All points are inside bbox
            return True
        
        return False
    
    def line_intersects(self, line1, line2):
        """Check if two line segments intersect"""
        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1

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

                    frame, boxes, scores, class_ids = self.model.predict(frame)
                    
                    for box, score, class_id in zip(boxes, scores, class_ids):
                        label = self.model.labels[class_id]
                        if label == 'person':
                            # Check if person bbox intersects with any ROI
                            intersects_roi = False
                            for roi in self.current_safety_rois:
                                if self.bbox_intersects_roi(box, roi['coords']):
                                    intersects_roi = True
                                    break
                            
                            # Set color based on ROI intersection
                            color = (0, 0, 255) if intersects_roi else (0, 255, 0)  # Red if intersects ROI, Green if not
                            x1, y1, x2, y2 = box.astype(int)

                            frame = self.model.plot_one_box(frame, x1, y1, x2, y2, score, label, color)

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