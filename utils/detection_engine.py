import cv2
import numpy as np
import os
import time
import re
from pyzbar import pyzbar
from openvino.runtime import Core
from ultralytics import YOLO


class DetectionEngine:
    """Handles all model inference and processing operations using OpenVINO."""
    
    def __init__(self, person_model_path, helmet_model_path, qr_model_path, device="cpu", debug_qr=False):
        """
        Initialize the detection engine with OpenVINO optimization.
        
        Args:
            person_model_path: Path to YOLO person detection model (.pt file)
            helmet_model_path: Path to YOLO helmet detection model (.pt file)
            qr_model_path: Path to YOLO QR code detection model (.pt file)
            device: Device to run inference on (cpu/cuda/gpu)
            debug_qr: Enable QR code debugging output
        """
        self.device = device
        self.debug_qr = debug_qr
        self.core = Core()
        self.use_openvino = True  # Flag to control which models to use
        
        # Load YOLO models from paths
        print("Loading YOLO models...")
        person_yolo = YOLO(person_model_path)
        helmet_yolo = YOLO(helmet_model_path)
        qr_yolo = YOLO(qr_model_path)
        
        # Get labels from original models
        self.person_labels = person_yolo.names
        self.helmet_labels = helmet_yolo.names
        self.qr_labels = qr_yolo.names
        
        # Convert and load person model to OpenVINO
        self.person_model_ov, self.person_input_height, self.person_input_width = self._load_openvino_model(
            person_model_path, person_yolo, "person"
        )
        
        # Convert and load helmet model to OpenVINO
        self.helmet_model_ov, self.helmet_input_height, self.helmet_input_width = self._load_openvino_model(
            helmet_model_path, helmet_yolo, "helmet"
        )
        
        # Convert and load QR model to OpenVINO
        self.qr_model_ov, self.qr_input_height, self.qr_input_width = self._load_openvino_model(
            qr_model_path, qr_yolo, "qr"
        )
        
        # Check if all OpenVINO models loaded successfully
        if all([self.person_model_ov, self.helmet_model_ov, self.qr_model_ov]):
            print("✓ Using OpenVINO models for inference")
            self.use_openvino = True
            self.person_model = self.person_model_ov
            self.helmet_model = self.helmet_model_ov
            self.qr_model = self.qr_model_ov
        else:
            print("⚠ Falling back to YOLO models for inference")
            self.use_openvino = False
            self.person_model = person_yolo
            self.helmet_model = helmet_yolo
            self.qr_model = qr_yolo
            
    def _load_openvino_model(self, model_path, yolo_model, model_name):
        """
        Convert YOLO model to OpenVINO format and load it.
        
        Args:
            model_path: Path to the original YOLO model file
            yolo_model: Loaded YOLO model object
            model_name: Name identifier for the model
            
        Returns:
            Tuple of (compiled_model, input_height, input_width)
        """
        # Construct OpenVINO model path
        model_basename = os.path.basename(model_path).split('.')[0]
        model_dirname = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
        openvino_dir = os.path.join(model_dirname, model_basename + "_openvino_model")
        xml_path = os.path.join(openvino_dir, model_basename + ".xml")
        
        # Export to OpenVINO if .xml doesn't exist
        if not os.path.exists(xml_path):
            print(f"  Exporting {model_name} model to OpenVINO format...")
            try:
                yolo_model.export(format="openvino", dynamic=False, half=True)
                print(f"  ✓ Exported to {openvino_dir}")
            except Exception as e:
                print(f"  ⚠ Warning: Could not export to OpenVINO: {e}")
                print(f"  Continuing with YOLO model...")
                return None, None, None
       
        try:
            # Load the OpenVINO model
            ov_model = self.core.read_model(xml_path)
            
            # Get input dimensions
            input_layer = ov_model.input(0)
            input_height = input_layer.shape[2]
            input_width = input_layer.shape[3]
            
            # Reshape model
            ov_model.reshape({0: [1, 3, input_height, input_width]})
            
            # Compile model for the target device
            compiled_model = self.core.compile_model(ov_model, self.device.upper())
                        
            return compiled_model, input_height, input_width
        except Exception as e:
            print(f"  ⚠ Warning: Could not load OpenVINO model: {e}")
            print(f"  Continuing with YOLO model...")
            return None, None, None
    
    def predict(self, model, image, conf=0.5):
        """
        Run model inference on an image using OpenVINO or YOLO.
        
        Args:
            model: Model to use for inference (OpenVINO or YOLO)
            image: Input image (numpy array)
            conf: Confidence threshold for detections
            
        Returns:
            Tuple of (boxes, scores, class_ids) as numpy arrays
        """
        try:
            if image is None:
                return np.array([]), np.array([]), np.array([])
            
            # Check if using OpenVINO model
            is_openvino = hasattr(model, 'input') or (self.use_openvino and model in [
                self.person_model_ov, self.helmet_model_ov, self.qr_model_ov
            ])
            
            # Use OpenVINO inference if available
            if is_openvino and self.use_openvino:
                return self._predict_openvino(model, image, conf)
            else:
                # Use YOLO inference
                return self._predict_yolo(model, image, conf)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _predict_yolo(self, model, image, conf=0.5):
        """Run YOLO inference (original method)"""
        results = model(image, device=self.device, verbose=False, conf=conf)
        
        boxes = []
        scores = []
        class_ids = []
        
        if len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        return np.array(boxes), np.array(scores), np.array(class_ids)
    
    def _predict_openvino(self, model, image, conf=0.5):
        """Run OpenVINO inference"""
        # Get input dimensions from model
        if model == self.person_model_ov:
            input_h, input_w = self.person_input_height, self.person_input_width
        elif model == self.helmet_model_ov:
            input_h, input_w = self.helmet_input_height, self.helmet_input_width
        elif model == self.qr_model_ov:
            input_h, input_w = self.qr_input_height, self.qr_input_width
        else:
            input_h, input_w = 640, 640  # Default
        
        # Preprocess image (now returns scaling info)
        orig_h, orig_w = image.shape[:2]
        input_tensor, scale, pad_x, pad_y = self._preprocess_image(image, input_h, input_w)
        
        # Run inference using OpenVINO API
        infer_request = model.create_infer_request()
        infer_request.infer({0: input_tensor})
        output = infer_request.get_output_tensor(0).data
        
        # Postprocess results (now uses scaling info)
        boxes, scores, class_ids = self._postprocess_openvino(
            output, orig_w, orig_h, input_w, input_h, conf, scale, pad_x, pad_y
        )
        
        return boxes, scores, class_ids
    
    def _preprocess_image(self, image, target_h, target_w):
        """Preprocess image for OpenVINO inference"""
        # Resize image while maintaining aspect ratio
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Convert to tensor format: (1, 3, H, W), normalized to [0, 1]
        input_tensor = padded.transpose(2, 0, 1)  # HWC -> CHW
        input_tensor = input_tensor.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        # Return tensor and scaling info for postprocessing
        return input_tensor, scale, pad_x, pad_y
    
    def _postprocess_openvino(self, output, orig_w, orig_h, input_w, input_h, conf_threshold, scale, pad_x, pad_y):
        """Postprocess OpenVINO output to extract boxes, scores, and class IDs"""
        # Output shape is typically (1, 84, 8400) for YOLO v8/v11
        # Where 84 = 4 (bbox) + 80 (classes)
        predictions = output[0].T  # Transpose to (8400, 84)
        
        # Extract boxes and scores
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        
        # Get max class scores and IDs
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # Filter by confidence threshold
        mask = max_scores > conf_threshold
        boxes_xywh = boxes_xywh[mask]
        scores = max_scores[mask]
        class_ids = class_ids[mask]
        
        if len(boxes_xywh) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert xywh to xyxy
        boxes = np.zeros_like(boxes_xywh)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
        
        # Remove padding offset
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        
        # Scale boxes to original image size
        boxes[:, [0, 2]] /= scale
        boxes[:, [1, 3]] /= scale
        
        # Clip boxes to image boundaries
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        
        # Apply NMS
        indices = self._nms(boxes, scores, iou_threshold=0.45)
        
        return boxes[indices], scores[indices], class_ids[indices]
    
    def _nms(self, boxes, scores, iou_threshold=0.45):
        """Non-maximum suppression"""
        if len(boxes) == 0:
            return np.array([])
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        keep_indices = []
        
        while len(sorted_indices) > 0:
            # Pick the box with highest score
            idx = sorted_indices[0]
            keep_indices.append(idx)
            
            if len(sorted_indices) == 1:
                break
            
            # Compute IoU with remaining boxes
            ious = self._compute_iou(boxes[idx], boxes[sorted_indices[1:]])
            
            # Keep boxes with IoU less than threshold
            mask = ious < iou_threshold
            sorted_indices = sorted_indices[1:][mask]
        
        return np.array(keep_indices)
    
    def _compute_iou(self, box, boxes):
        """Compute IoU between one box and multiple boxes"""
        # Get coordinates
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        # Compute intersection area
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        
        # Compute IoU
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def detect_helmet_on_person(self, image, person_bbox, conf=0.5):
        """
        Detect helmet on a person by cropping the person region and running helmet detection.
        
        Args:
            image: Full image (numpy array)
            person_bbox: Bounding box of the person [x1, y1, x2, y2]
            conf: Confidence threshold for helmet detection
            
        Returns:
            Tuple of (helmet_box, helmet_score) where helmet_box is [x1, y1, x2, y2] 
            in absolute coordinates, or (None, None) if no helmet detected
        """
        try:
            x1, y1, x2, y2 = person_bbox
            person_crop = image[int(y1):int(y2), int(x1):int(x2)]
            
            boxes, scores, class_ids = self.predict(self.helmet_model, person_crop, conf=conf)
            
            if len(boxes) > 0:
                max_idx = np.argmax(scores)
                helmet_box = boxes[max_idx]
                hx1, hy1, hx2, hy2 = helmet_box
                
                # Convert to absolute coordinates
                hx1 = int(x1 + hx1)
                hy1 = int(y1 + hy1)
                hx2 = int(x1 + hx2)
                hy2 = int(y1 + hy2)
                
                return [hx1, hy1, hx2, hy2], scores[max_idx]
            
            return None, None
            
        except Exception as e:
            return None, None
    
    def detect_qr_code_on_helmet(self, image, helmet_bbox, conf=0.5):
        """
        Detect QR code on a helmet and decode it.
        
        Args:
            image: Full image (numpy array)
            helmet_bbox: Bounding box of the helmet [x1, y1, x2, y2]
            conf: Confidence threshold for QR detection
            
        Returns:
            Tuple of (qr_box, qr_score, qr_data) where qr_box is [x1, y1, x2, y2]
            in absolute coordinates, or (None, None, None) if no QR code detected
        """
        try:
            x1, y1, x2, y2 = helmet_bbox
            helmet_crop = image[int(y1):int(y2), int(x1):int(x2)]

            boxes, scores, class_ids = self.predict(self.qr_model, helmet_crop, conf=conf)
            
            qr_data = None
            
            if len(boxes) > 0:
                max_idx = np.argmax(scores)
                qr_box = boxes[max_idx]
                qx1, qy1, qx2, qy2 = qr_box
                
                # Add small padding to capture the QR code
                qx1_abs = int(x1 + qx1) - 10
                qy1_abs = int(y1 + qy1) - 10
                qx2_abs = int(x1 + qx2) + 10
                qy2_abs = int(y1 + qy2) + 10
                
                padding = 0
                h_img, w_img = image.shape[:2]
                qy1_padded = max(0, qy1_abs - padding)
                qy2_padded = min(h_img, qy2_abs + padding)
                qx1_padded = max(0, qx1_abs - padding)
                qx2_padded = min(w_img, qx2_abs + padding)
                
                qr_crop = image[qy1_padded:qy2_padded, qx1_padded:qx2_padded]
               
                decoded_objects = self._try_decode_with_preprocessing(qr_crop)
                
                if decoded_objects:
                    qr_data = decoded_objects[0].data.decode('utf-8')
                    if self.debug_qr:
                        print(f"[DEBUG] QR Code decoded from helmet - Raw data length: {len(qr_data)} chars")
                
                return [qx1_abs, qy1_abs, qx2_abs, qy2_abs], scores[max_idx], qr_data
            
            return None, None, None
            
        except Exception as e:
            return None, None, None
    
    def _try_decode_with_preprocessing(self, image):
        """
        Try to decode QR code with various preprocessing techniques.
        
        Args:
            image: QR code crop image (numpy array)
            
        Returns:
            Decoded QR code objects list or None if decoding failed
        """
        # Save original QR crop for debugging
        if self.debug_qr:
            timestamp = time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            debug_dir = "/home/claude/qr_debug"
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(f"{debug_dir}/qr_crop_original_{timestamp}.jpg", image)
            print(f"[DEBUG] Saved QR crop to {debug_dir}/qr_crop_original_{timestamp}.jpg - Size: {image.shape}")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Try a few simple preprocessing methods
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        images_to_try = [
            ('gray', gray),
            ('otsu', binary_otsu)
        ]
        
        # Try with minimal rotation angles
        angles = [0, -5, 5, -10, 10]
        
        for name, img in images_to_try:
            for angle in angles:
                if angle == 0:
                    test_img = img
                else:
                    h, w = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    
                    cos_val = np.abs(M[0, 0])
                    sin_val = np.abs(M[0, 1])
                    new_w = int((h * sin_val) + (w * cos_val))
                    new_h = int((h * cos_val) + (w * sin_val))
                    
                    M[0, 2] += (new_w / 2) - center[0]
                    M[1, 2] += (new_h / 2) - center[1]
                    
                    test_img = cv2.warpAffine(img, M, (new_w, new_h), 
                                            borderMode=cv2.BORDER_CONSTANT, 
                                            borderValue=255)
                
                decoded = pyzbar.decode(test_img)
                if decoded:
                    if self.debug_qr:
                        print(f"✓ QR Code decoded successfully with: method={name}, angle={angle}")
                    return decoded
        
        if self.debug_qr:
            print(f"✗ Failed to decode QR code")
        
        return None
    
    def parse_qr_data(self, qr_text):
        """
        Parse QR code text to extract ID and Niveau information.
        
        Args:
            qr_text: Raw QR code text string
            
        Returns:
            Formatted string "ID : Niveau" or None if parsing failed
        """
        try:
            if self.debug_qr:
                print(f"[DEBUG] Raw QR Code Data: '{qr_text}'")
            
            # Try the new format: ID:Niveau
            # Example: "ABC123:N2" or "12345:Level3"
            simple_match = re.search(r'^([^:]+):([^:]+)$', qr_text.strip())
            if simple_match:
                id_value = simple_match.group(1).strip()
                niveau_value = simple_match.group(2).strip()
                result = f"{id_value} : {niveau_value}"
                if self.debug_qr:
                    print(f"[DEBUG] Parsed QR (simple format) - ID: {id_value}, Niveau: {niveau_value}")
                    print(f"[DEBUG] Result: {result}")
                return result
            
            # Fallback: Try old format with "Matricule" and "Niveau" keywords
            matricule = None
            niveau = None
            
            matricule_match = re.search(r'(?:Matricule|ID)\s*:\s*(\S+)', qr_text, re.IGNORECASE)
            if matricule_match:
                matricule = matricule_match.group(1)
                if self.debug_qr:
                    print(f"[DEBUG] Found Matricule/ID: {matricule}")
            
            niveau_match = re.search(r'Niveau\s*:\s*(\S+)', qr_text, re.IGNORECASE)
            if niveau_match:
                niveau = niveau_match.group(1)
                if self.debug_qr:
                    print(f"[DEBUG] Found Niveau: {niveau}")
            
            if matricule and niveau:
                result = f"{matricule} : {niveau}"
                if self.debug_qr:
                    print(f"[DEBUG] Parsed QR result (old format): {result}")
                return result
            
            if self.debug_qr:
                print(f"[DEBUG] Failed to parse - Matricule: {matricule}, Niveau: {niveau}")
                print(f"[DEBUG] QR text doesn't match expected format. Raw text: '{qr_text}'")
            return None
        except Exception as e:
            if self.debug_qr:
                print(f"[DEBUG] Exception in parse_qr_data: {e}")
            return None
    
    @staticmethod
    def plot_one_box(image, x1, y1, x2, y2, color):
        """
        Draw a bounding box on an image.
        
        Args:
            image: Image to draw on (numpy array)
            x1, y1, x2, y2: Bounding box coordinates
            color: BGR color tuple
            
        Returns:
            Image with bounding box drawn
        """
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        return image
