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
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

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
        self.high_violation = False
        self.low_violation = False

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
            # Check if we're in config mode (marker.html) by looking at the referer
            config_mode = request.headers.get('Referer', '').endswith('/config')
            return Response(self.video_stream(config_mode), mimetype='multipart/x-mixed-replace; boundary=frame')

        @app.route('/zone_status')
        def zone_status():
            """Return current zone violation status for audio alerts"""
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

    def is_in_roi(self, bbox, roi):
        """Legacy method for center point checking (kept for compatibility)"""
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        try:
            roi_polygon = Polygon(roi)
            if not roi_polygon.is_valid:
                return False
            from shapely.geometry import Point
            center_point = Point(center_x, center_y)
            return roi_polygon.contains(center_point)
        except Exception:
            return False
    
    def bbox_intersects_roi(self, bbox, roi):
        """Check if bounding box intersects with ROI polygon using shapely with pure Python fallback"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are in the correct order (min, max)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            # Create shapely polygon from ROI coordinates
            if len(roi) < 3:
                return False
                
            # Ensure we have valid coordinate pairs
            roi_coords = []
            for coord in roi:
                if isinstance(coord, (list, tuple)) and len(coord) == 2:
                    roi_coords.append((float(coord[0]), float(coord[1])))
                else:
                    return False
            
            roi_polygon = Polygon(roi_coords)
            
            # Check if polygon is valid
            if not roi_polygon.is_valid:
                # Try to fix invalid polygon
                roi_polygon = roi_polygon.buffer(0)
                if not roi_polygon.is_valid:
                    # Fallback to pure Python method if shapely fails
                    return self._python_bbox_intersects_roi(bbox, roi)
            
            # Create bounding box as shapely rectangle
            bbox_polygon = box(x_min, y_min, x_max, y_max)
            
            # Check for intersection
            return roi_polygon.intersects(bbox_polygon)
            
        except Exception as e:
            # Fallback to pure Python method if shapely fails
            return self._python_bbox_intersects_roi(bbox, roi)
    
    def _python_bbox_intersects_roi(self, bbox, roi):
        """Pure Python fallback method for bbox-ROI intersection"""
        try:
            x1, y1, x2, y2 = bbox
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            # Check if any corner of the bbox is inside the polygon
            corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            for corner in corners:
                if self._point_in_polygon(corner, roi):
                    return True
            
            # Check if any ROI point is inside the bbox
            for point in roi:
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    px, py = point[0], point[1]
                    if x_min <= px <= x_max and y_min <= py <= y_max:
                        return True
            
            # Check if any bbox edge intersects with any ROI edge
            bbox_edges = [
                ((x_min, y_min), (x_max, y_min)),  # top edge
                ((x_max, y_min), (x_max, y_max)),  # right edge
                ((x_max, y_max), (x_min, y_max)),  # bottom edge
                ((x_min, y_max), (x_min, y_min))   # left edge
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
        """Ray casting algorithm to check if point is inside polygon"""
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
        """Check if two line segments intersect"""
        (x1, y1), (x2, y2) = seg1
        (x3, y3), (x4, y4) = seg2
        
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw((x1, y1), (x3, y3), (x4, y4)) != ccw((x2, y2), (x3, y3), (x4, y4)) and \
               ccw((x1, y1), (x2, y2), (x3, y3)) != ccw((x1, y1), (x2, y2), (x4, y4))

    def video_stream(self, config_mode=False):
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
                    
                    # Only draw ROIs on the server side if NOT in config mode
                    if not config_mode:
                        for roi in self.current_safety_rois:
                            points = np.array(roi['coords'], np.int32).reshape(-1, 1, 2)
                            # Use different colors based on zone type
                            zone_type = roi.get('type', 'high')  # Default to 'high'
                            if zone_type == 'high':
                                roi_color = (0, 0, 255)  # Red for high security
                            else:  # 'low'
                                roi_color = (0, 165, 255)  # Orange for low security
                            
                            cv2.polylines(frame, [points], True, roi_color, 2)

                    frame, boxes, scores, class_ids = self.model.predict(frame)
                    
                    # Reset violation flags
                    self.high_violation = False
                    self.low_violation = False
                    
                    for box, score, class_id in zip(boxes, scores, class_ids):
                        label = self.model.labels[class_id]
                        if label == 'person':
                            # First check high security zones
                            in_high_security = False
                            for roi in self.current_safety_rois:
                                if roi.get('type', 'high') == 'high' and self.bbox_intersects_roi(box, roi['coords']):
                                    in_high_security = True
                                    self.high_violation = True
                                    break
                            
                            # Only check low security zones if not in high security
                            in_low_security = False
                            if not in_high_security:
                                for roi in self.current_safety_rois:
                                    if roi.get('type', 'high') == 'low' and self.bbox_intersects_roi(box, roi['coords']):
                                        in_low_security = True
                                        self.low_violation = True
                                        break
                            
                            # Set color based on zone type
                            if in_high_security:
                                color = (0, 0, 255)  # Red for high security violation
                            elif in_low_security:
                                color = (0, 165, 255)  # Orange for low security violation
                            else:
                                color = (0, 255, 0)  # Green if not in any zone
                            
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