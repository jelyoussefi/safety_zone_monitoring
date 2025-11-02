import os
import argparse
import fire
import cv2
import numpy as np
import time
import json
import threading
import logging
from threading import Thread, Condition
from flask import Flask, redirect, url_for, render_template, make_response, jsonify, request, Response
from flask_bootstrap import Bootstrap
from flask_restful import Resource, Api, reqparse
from utils.detection_engine import DetectionEngine
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
import math
from pyzbar import pyzbar
import re

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class SafetyZoneMonitor:
    def __init__(self, det_person, det_helmet, input, config_file, device="CPU", debug_qr=False, det_qr_code=None):
        with open(config_file) as f:
            self.config = json.load(f)
        
        self.device = device.lower() if device.upper() == 'CPU' else device
        self.debug_qr = debug_qr
        
        try:
            # Create DetectionEngine instance with model paths
            self.detection_engine = DetectionEngine(det_person, det_helmet, det_qr_code, device=self.device, debug_qr=debug_qr)

        except Exception as e:
            print(f"✗ Error loading models: {e}")
            raise
        
        self.input = input
        self.cap = cv2.VideoCapture(input)
       
        self.config_file = config_file
        self.app = Flask(__name__)
        self.port = 80
        self.running = False
        self.cv = Condition()
        self.current_safety_rois = self.config.get('safety_rois', [])
        self.frame = None
        self.high_violation = False
        self.low_violation = False

    def _resize_with_aspect_ratio(self, image, target_width=1280, target_height=720):
        h, w = image.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        return resized

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
            config_mode = request.headers.get('Referer', '').endswith('/config')
            return Response(self.video_stream(config_mode), mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/zone_status')
        def zone_status():
            return jsonify({
                'high_violation': getattr(self, 'high_violation', False),
                'low_violation': getattr(self, 'low_violation', False)
            })

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
    
    def bbox_intersects_roi(self, bbox, roi):
        try:
            x1, y1, x2, y2 = bbox
            
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            if len(roi) < 3:
                return False
                
            roi_coords = []
            for coord in roi:
                if isinstance(coord, (list, tuple)) and len(coord) == 2:
                    roi_coords.append((float(coord[0]), float(coord[1])))
                else:
                    return False
            
            roi_polygon = Polygon(roi_coords)
            
            if not roi_polygon.is_valid:
                roi_polygon = roi_polygon.buffer(0)
                if not roi_polygon.is_valid:
                    return self._python_bbox_intersects_roi(bbox, roi)
            
            bbox_polygon = box(x_min, y_min, x_max, y_max)
            
            return roi_polygon.intersects(bbox_polygon)
            
        except Exception as e:
            return self._python_bbox_intersects_roi(bbox, roi)
    
    def _python_bbox_intersects_roi(self, bbox, roi):
        try:
            x1, y1, x2, y2 = bbox
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            for corner in corners:
                if self._point_in_polygon(corner, roi):
                    return True
            
            for point in roi:
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    px, py = point[0], point[1]
                    if x_min <= px <= x_max and y_min <= py <= y_max:
                        return True
            
            bbox_edges = [
                ((x_min, y_min), (x_max, y_min)),
                ((x_max, y_min), (x_max, y_max)),
                ((x_max, y_max), (x_min, y_max)),
                ((x_min, y_max), (x_min, y_min))
            ]
            
            for i in range(len(roi)):
                roi_edge = (roi[i], roi[(i + 1) % len(roi)])
                for bbox_edge in bbox_edges:
                    if self._line_segments_intersect(bbox_edge, roi_edge):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _point_in_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _line_segments_intersect(self, seg1, seg2):
        (x1, y1), (x2, y2) = seg1
        (x3, y3), (x4, y4) = seg2
        
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw((x1, y1), (x3, y3), (x4, y4)) != ccw((x2, y2), (x3, y3), (x4, y4)) and \
               ccw((x1, y1), (x2, y2), (x3, y3)) != ccw((x1, y1), (x2, y2), (x4, y4))

    def video_stream(self, config_mode=False):
        self.running = True
        self.cap = cv2.VideoCapture(self.input)
        
        while self.running:
            with self.cv:
                ret, self.frame = self.cap.read()
                
                # If end of video is reached, restart from beginning
                if not ret or self.frame is None:
                    # Check if it's a video file (not a camera stream)
                    if isinstance(self.input, str) and not self.input.isdigit():
                        # Release current capture and restart
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.input)
                        ret, self.frame = self.cap.read()
                        
                        # If still can't read, break
                        if not ret or self.frame is None:
                            break
                    else:
                        # For camera streams, just break
                        break

                frame = self.frame.copy()
               
                if not config_mode:
                    for roi in self.current_safety_rois:
                        points = np.array(roi['coords'], np.int32).reshape(-1, 1, 2)
                        zone_type = roi.get('type', 'high')
                        if zone_type == 'high':
                            roi_color = (0, 0, 255)
                        else:
                            roi_color = (0, 165, 255)
                        
                        cv2.polylines(frame, [points], True, roi_color, 2)

                boxes, scores, class_ids = self.detection_engine.predict(self.detection_engine.person_model, frame, conf=0.5)
                
                self.high_violation = False
                self.low_violation = False
                
                for idx, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                    label = self.detection_engine.person_labels[class_id]
                    if label == 'person':
                        helmet_box, helmet_score = self.detection_engine.detect_helmet_on_person(frame, box, conf=0.5)
                        has_helmet = helmet_box is not None
                        
                        qr_box = None
                        qr_score = None
                        qr_data = None
                        parsed_qr = None
                        
                        if helmet_box:
                            qr_box, qr_score, qr_data = self.detection_engine.detect_qr_code_on_helmet(frame, helmet_box, conf=0.5)
                            if qr_data:
                                parsed_qr = self.detection_engine.parse_qr_data(qr_data)
                            
                        in_zone = False
                        in_high_security_zone = False
                        
                        for roi in self.current_safety_rois:
                            if self.bbox_intersects_roi(box, roi['coords']):
                                in_zone = True
                                zone_type = roi.get('type', 'high')
                                
                                if zone_type == 'high':
                                    in_high_security_zone = True
                                    if not has_helmet:
                                        self.high_violation = True
                                else:
                                    if not has_helmet:
                                        self.low_violation = True
                                break
                        
                        # Extract level from parsed QR data
                        qr_level = None
                        if parsed_qr:
                            # Assuming parsed_qr contains level like "H1" or "H2"
                            # If it's a full string, try to extract H1 or H2
                            if "H2" in parsed_qr.upper():
                                qr_level = "H2"
                            elif "H1" in parsed_qr.upper():
                                qr_level = "H1"
                        
                        # Determine bounding box color based on new logic
                        # Green if: has_helmet AND (level is H2 OR (level is H1 AND NOT in high security zone))
                        # Red otherwise
                        if has_helmet and (qr_level == "H2" or (qr_level == "H1" and not in_high_security_zone)):
                            color = (0, 255, 0)  # Green
                        else:
                            color = (0, 0, 255)  # Red
                        
                        x1, y1, x2, y2 = [round(coord) for coord in box]
                        frame = self.detection_engine.plot_one_box(frame, x1, y1, x2, y2, color)
                        
                        if helmet_box is not None:
                            hx1, hy1, hx2, hy2 = [round(coord) for coord in helmet_box]
                            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
                        
                        if qr_box is not None:
                            qx1, qy1, qx2, qy2 = [round(x) for x in qr_box]
                            cv2.rectangle(frame, (qx1, qy1), (qx2, qy2), (255, 0, 255), 3)
                        
                        if parsed_qr:
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.8
                            thickness = 2
                            
                            (text_width, text_height), _ = cv2.getTextSize(parsed_qr, font, font_scale, thickness)
                            
                            text_x = center_x - text_width // 2
                            text_y = center_y + text_height // 2
                            
                            padding = 5
                            cv2.rectangle(frame, 
                                        (text_x - padding, text_y - text_height - padding),
                                        (text_x + text_width + padding, text_y + padding),
                                        (0, 0, 0), -1)
                            
                            cv2.putText(frame, parsed_qr, (text_x, text_y), 
                                       font, font_scale, (255, 255, 255), thickness)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.025)

        with self.cv:
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self.running = False

def main(det_person, det_helmet, input, config, device, debug_qr=False, det_qr_code=None, **kwargs):
    szm = SafetyZoneMonitor(det_person, det_helmet, input, config, device, debug_qr=debug_qr, det_qr_code=det_qr_code)
    szm.run()

if __name__ == "__main__":
    fire.Fire(main)
