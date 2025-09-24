# yolo_thread.py - YOLO detection and tracking thread (Fixed version)

import os
import cv2
import numpy as np
import time
import queue
import logging
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
import colorsys

# TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Warning: TensorRT not available, YOLO detection disabled")

# Import JPDAF tracker
try:
    from jpdaf_tracker import JPDAFilter
    JPDAF_AVAILABLE = True
except ImportError:
    JPDAF_AVAILABLE = False
    print("Warning: JPDAF tracker not available, tracking disabled")

logger = logging.getLogger(__name__)


class TensorRTInference:
    """TensorRT inference engine for YOLO"""
    def __init__(self, engine_path):
        """Initialize TensorRT inference engine"""
        # Initialize CUDA context in this thread
        import pycuda.autoinit
        self.cuda_context = pycuda.autoinit.context
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        logger.info("Loading TensorRT engine...")
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT context")
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        # Setup bindings based on TensorRT version
        self._setup_bindings()
        
        logger.info("TensorRT engine loaded successfully")
        logger.info(f"Input shape: {self.inputs[0]['shape']}")
        logger.info(f"Output count: {len(self.outputs)}")
    
    def _setup_bindings(self):
        """Setup input/output bindings"""
        try:
            # TensorRT 8+ API
            num_io_tensors = self.engine.num_io_tensors
            tensor_names = [self.engine.get_tensor_name(i) for i in range(num_io_tensors)]
            
            for name in tensor_names:
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = self.engine.get_tensor_shape(name)
                
                if -1 in shape:
                    if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        shape = (1, 3, 640, 640)
                        self.context.set_input_shape(name, shape)
                
                size = trt.volume(shape)
                device_mem = cuda.mem_alloc(size * dtype().itemsize)
                self.bindings.append(int(device_mem))
                host_mem = cuda.pagelocked_empty(size, dtype)
                
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.inputs.append({'name': name, 'shape': shape, 'dtype': dtype, 
                                      'host': host_mem, 'device': device_mem})
                else:
                    self.outputs.append({'name': name, 'shape': shape, 'dtype': dtype, 
                                       'host': host_mem, 'device': device_mem})
                    
        except AttributeError:
            # TensorRT 7.x API
            logger.info("Using TensorRT 7.x API")
            num_bindings = self.engine.num_bindings
            
            for i in range(num_bindings):
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = self.engine.get_binding_shape(i)
                size = trt.volume(shape)
                
                device_mem = cuda.mem_alloc(size * dtype().itemsize)
                self.bindings.append(int(device_mem))
                host_mem = cuda.pagelocked_empty(size, dtype)
                
                if self.engine.binding_is_input(i):
                    self.inputs.append({'name': name, 'shape': shape, 'dtype': dtype, 
                                      'host': host_mem, 'device': device_mem})
                else:
                    self.outputs.append({'name': name, 'shape': shape, 'dtype': dtype, 
                                       'host': host_mem, 'device': device_mem})
    
    def preprocess(self, img):
        """Preprocess image for inference"""
        if img is None:
            raise ValueError("Cannot process empty image")
        
        self.original_shape = img.shape[:2]
        
        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 640x640
        img_resized = cv2.resize(img, (640, 640))
        
        # Normalize to [0,1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to CHW format
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension
        img_batch = np.expand_dims(img_chw, axis=0)
        
        # Ensure contiguous memory
        img_batch = np.ascontiguousarray(img_batch)
        
        # Convert to model dtype
        if self.inputs[0]['dtype'] != img_batch.dtype:
            img_batch = img_batch.astype(self.inputs[0]['dtype'])
            
        return img_batch
    
    def infer(self, input_data):
        """Execute inference"""
        # Push CUDA context
        self.cuda_context.push()
        
        try:
            # Ensure correct shape
            expected_size = np.prod(self.inputs[0]['shape'])
            input_size = input_data.size
            
            if input_size != expected_size:
                input_data = input_data.reshape(self.inputs[0]['shape'])
            
            if input_data.dtype != self.inputs[0]['dtype']:
                input_data = input_data.astype(self.inputs[0]['dtype'])
            
            # Copy to device
            np.copyto(self.inputs[0]['host'], input_data.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            # Execute
            try:
                for inp in self.inputs:
                    self.context.set_tensor_address(inp['name'], inp['device'])
                for out in self.outputs:
                    self.context.set_tensor_address(out['name'], out['device'])
                self.context.execute_async_v3(stream_handle=self.stream.handle)
            except AttributeError:
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # Copy output
            for output in self.outputs:
                cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
            
            self.stream.synchronize()
            
            output = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
            return output
            
        finally:
            # Pop CUDA context
            self.cuda_context.pop()
    
    def postprocess(self, output, conf_threshold=0.25, iou_threshold=0.45):
        """Postprocess YOLO output"""
        if output.ndim == 3:
            output = output[0]
        
        predictions = output.T
        boxes = predictions[:, :4]  # x, y, w, h
        scores = predictions[:, 4:]  # class scores
        
        class_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        mask = class_scores > conf_threshold
        if not mask.any():
            return [], [], []
        
        boxes = boxes[mask]
        class_scores = class_scores[mask]
        class_ids = class_ids[mask]
        
        # Convert to xyxy format
        boxes_xyxy = self.xywh2xyxy(boxes)
        
        # Scale to original size
        scale_x = self.original_shape[1] / 640
        scale_y = self.original_shape[0] / 640
        
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y
        
        # NMS
        indices = self.nms(boxes_xyxy, class_scores, iou_threshold)
        
        return boxes_xyxy[indices], class_scores[indices], class_ids[indices]
    
    def xywh2xyxy(self, boxes):
        """Convert box format from xywh to xyxy"""
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return boxes_xyxy
    
    def nms(self, boxes, scores, iou_threshold):
        """Non-maximum suppression"""
        if len(boxes) == 0:
            return []
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Free CUDA memory
            for inp in self.inputs:
                inp['device'].free()
            for out in self.outputs:
                out['device'].free()
            
            # Destroy context
            if hasattr(self, 'cuda_context'):
                self.cuda_context.pop()
                self.cuda_context.detach()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


class YOLODetectionThread(QThread):
    """Thread for YOLO detection and tracking"""
    detection_result_signal = pyqtSignal(np.ndarray, list, list, list, dict)  # image, boxes, scores, class_ids, tracks
    status_signal = pyqtSignal(str, str)  # message, type
    fps_signal = pyqtSignal(float, float)  # detection_fps, tracking_fps
    
    def __init__(self, engine_path="weights/yolov8-weights/best.engine"):
        super().__init__()
        self.engine_path = engine_path
        self.running = False
        self.detecting = False
        self.mutex = QMutex()
        self.frame_queue = queue.Queue(maxsize=5)
        self.engine = None
        self.tracker = None
        self.skip_frames = 0
        self.frame_counter = 0
        
        # Performance monitoring
        self.last_detection_time = time.time()
        self.last_tracking_time = time.time()
        self.detection_times = []
        self.tracking_times = []
        
        # Class names
        self.class_names = [
            'ball_contact',
            'ball_free', 
            'ball_inside',
            'ball_under',
            'pipette_tip'
        ]
        
        # Track colors
        self.track_colors = self._generate_colors(100)
    
    def _generate_colors(self, num_colors):
        """Generate distinct colors for tracks"""
        colors = []
        for i in range(num_colors):
            h = i / num_colors
            s = 0.8
            v = 0.8
            rgb = colorsys.hsv_to_rgb(h, s, v)
            rgb = tuple(int(x * 255) for x in rgb)
            colors.append(rgb)
        return colors
    
    def initialize(self):
        """Initialize YOLO engine and tracker - called in the thread"""
        try:
            if not TENSORRT_AVAILABLE:
                self.status_signal.emit("TensorRT not available", "error")
                return False
            
            # Check if engine file exists
            if not os.path.exists(self.engine_path):
                self.status_signal.emit(f"Engine file not found: {self.engine_path}", "error")
                return False
            
            # Initialize TensorRT engine (this will create CUDA context in this thread)
            self.status_signal.emit("Initializing YOLO engine...", "info")
            self.engine = TensorRTInference(self.engine_path)
            
            # Warm up the engine
            self.status_signal.emit("Warming up YOLO engine...", "info")
            
            # Push context for warmup
            self.engine.cuda_context.push()
            try:
                dummy_input = np.random.randn(1, 3, 640, 640).astype(self.engine.inputs[0]['dtype'])
                for _ in range(5):
                    # Inference already handles context push/pop
                    output = self.engine.infer(dummy_input)
            finally:
                self.engine.cuda_context.pop()
            
            # Initialize tracker if available
            if JPDAF_AVAILABLE:
                self.tracker = JPDAFilter(
                    process_noise=20.0,
                    measure_noise=2.0,
                    detect_prob=0.7,
                    gate_prob=0.95
                )
                self.status_signal.emit("JPDAF tracker initialized", "info")
            else:
                self.status_signal.emit("JPDAF tracker not available", "warning")
            
            self.status_signal.emit("YOLO detection ready", "info")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO: {e}")
            self.status_signal.emit(f"YOLO initialization failed: {str(e)}", "error")
            return False
    
    def run(self):
        """Main thread loop"""
        self.running = True
        
        # Initialize in the thread context
        if not self.initialize():
            self.running = False
            return
        
        while self.running:
            with QMutexLocker(self.mutex):
                should_detect = self.detecting
            
            if should_detect:
                try:
                    # Get frame with timeout
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # Process frame
                    self.process_frame(frame)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"YOLO processing error: {e}")
                    self.status_signal.emit(f"Detection error: {str(e)}", "error")
            else:
                time.sleep(0.1)
        
        # Cleanup
        if self.engine:
            self.engine.cleanup()
    
    def process_frame(self, frame):
        """Process a single frame for detection and tracking"""
        try:
            # Detection
            detection_start = time.time()
            
            # Preprocess
            input_data = self.engine.preprocess(frame)
            
            # Inference (handles context internally)
            output = self.engine.infer(input_data)
            
            # Postprocess
            boxes, scores, class_ids = self.engine.postprocess(output, conf_threshold=0.25, iou_threshold=0.45)
            
            detection_time = time.time() - detection_start
            self.detection_times.append(detection_time)
            
            # Tracking
            tracks_dict = {}
            if self.tracker and len(boxes) > 0:
                tracking_start = time.time()
                
                # Convert boxes to center points for tracking
                detection_points = []
                for box in boxes:
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    detection_points.append((x_center, y_center))
                
                # Update tracker
                self.tracker.predict()
                self.tracker.correct(detection_points)
                
                # Get active tracks
                active_tracks = self.tracker.get_active_tracks()
                for track in active_tracks:
                    if len(track.trajectory) >= 2:
                        tracks_dict[track.id] = {
                            'trajectory': track.trajectory,
                            'color': self.track_colors[track.id % len(self.track_colors)]
                        }
                
                tracking_time = time.time() - tracking_start
                self.tracking_times.append(tracking_time)
            
            # Calculate FPS
            if len(self.detection_times) > 10:
                self.detection_times = self.detection_times[-10:]
            if len(self.tracking_times) > 10:
                self.tracking_times = self.tracking_times[-10:]
            
            avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
            avg_tracking_time = np.mean(self.tracking_times) if self.tracking_times else 0
            
            detection_fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
            tracking_fps = 1.0 / (avg_detection_time + avg_tracking_time) if (avg_detection_time + avg_tracking_time) > 0 else 0
            
            # Emit results
            self.detection_result_signal.emit(frame, boxes, scores, class_ids, tracks_dict)
            self.fps_signal.emit(detection_fps, tracking_fps)
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
    
    def update_frame(self, frame):
        """Update frame for processing"""
        if not self.detecting:
            return
        
        # Frame skipping logic
        self.frame_counter += 1
        if self.frame_counter % (self.skip_frames + 1) != 0:
            return
        
        # Try to add frame to queue
        try:
            # Clear old frames if queue is full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put(frame.copy())
        except Exception as e:
            logger.error(f"Failed to update frame: {e}")
    
    def start_detection(self):
        """Start detection and tracking"""
        with QMutexLocker(self.mutex):
            self.detecting = True
            self.frame_counter = 0
            self.detection_times = []
            self.tracking_times = []
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        self.status_signal.emit("Detection and tracking started", "info")
    
    def stop_detection(self):
        """Stop detection and tracking"""
        with QMutexLocker(self.mutex):
            self.detecting = False
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset tracker
        if self.tracker:
            self.tracker = JPDAFilter(
                process_noise=20.0,
                measure_noise=2.0,
                detect_prob=0.7,
                gate_prob=0.95
            )
        
        self.status_signal.emit("Detection and tracking stopped", "info")
    
    def set_skip_frames(self, skip):
        """Set frame skipping rate"""
        self.skip_frames = max(0, skip)
    
    def stop(self):
        """Stop the thread"""
        self.running = False
        self.detecting = False
        self.wait(1000)