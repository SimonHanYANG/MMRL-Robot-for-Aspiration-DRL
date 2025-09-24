import sys
import os
import time
import queue
import logging
import numpy as np
import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QMutex, QTimer, pyqtSlot, QObject
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QKeyEvent
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
                            QPushButton, QHBoxLayout, QMessageBox, QGridLayout, QGroupBox, 
                            QLineEdit, QFormLayout, QSpinBox, QDoubleSpinBox)

# Import hardware control modules
from stage import Stage
from arm import armConnectState, armWorkingState, armGetMotorPos, armMovebyPos
from pump_thread import LeftPumpThread

from yolo_thread import YOLODetectionThread

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoRecorder(QObject):
    """Video recorder worker object that runs in a separate thread"""
    start_signal = pyqtSignal()
    stop_signal = pyqtSignal()

    def __init__(self, filename, fps, frame_size):
        super().__init__()
        self.filename = filename
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.start_signal.connect(self.start_recording)
        self.stop_signal.connect(self.stop_recording)
        self.isColor = False  # Grayscale by default
        self.logger = logging.getLogger(__name__)

    def start_recording(self):
        """Start video recording"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.filename, 
            fourcc, 
            self.fps, 
            self.frame_size,
            self.isColor
        )
        if not self.writer.isOpened():
            self.logger.error(f"Failed to open video file {self.filename}")

    @pyqtSlot(np.ndarray)
    def write_frame(self, frame):
        """Write a frame to the video file"""
        if self.writer and self.writer.isOpened():
            # Ensure frame is the correct size
            if (frame.shape[1], frame.shape[0]) != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            # Convert to grayscale if needed
            if len(frame.shape) == 3 and not self.isColor:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.writer.write(frame)

    def stop_recording(self):
        """Stop video recording and release resources"""
        if self.writer:
            self.writer.release()
            self.writer = None
            self.logger.info(f"Video saved: {self.filename}")


class VideoRecordingThread(QThread):
    """Thread for handling video recording"""
    status_signal = pyqtSignal(str, str)  # message, type
    recording_started_signal = pyqtSignal()
    recording_stopped_signal = pyqtSignal(str)  # filename
    
    def __init__(self):
        super().__init__()
        self.mutex = QMutex()
        self.is_recording = False
        self.video_recorder = None
        self.video_thread = None
        self.current_frame = None
        self.frame_queue = queue.Queue(maxsize=100)
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Create savedImg directory if it doesn't exist
        os.makedirs("savedImg", exist_ok=True)
    
    def run(self):
        """Main thread loop"""
        self.running = True
        while self.running:
            if self.is_recording:
                try:
                    # Get frame from queue with timeout
                    frame = self.frame_queue.get(timeout=0.1)
                    if self.video_recorder:
                        # Emit frame to video recorder
                        self.video_recorder.write_frame(frame)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error writing frame: {e}")
            else:
                time.sleep(0.1)  # Sleep when not recording
    
    def update_frame(self, image):
        """Update the current frame for recording"""
        if self.is_recording:
            try:
                # Add frame to queue, drop oldest if full
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(image.copy())
            except Exception as e:
                self.logger.error(f"Error updating frame: {e}")
    
    def start_recording(self, width=1600, height=1200, fps=30):
        """Start video recording"""
        self.mutex.lock()
        try:
            if self.is_recording:
                self.status_signal.emit("Already recording", "warning")
                return False
            
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"savedImg/capture_{timestamp}.mp4"
            
            # Create video recorder
            self.video_recorder = VideoRecorder(filename, fps, (width, height))
            
            # Start recording
            self.video_recorder.start_recording()
            self.is_recording = True
            
            # Clear frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.status_signal.emit(f"Recording started: {filename}", "info")
            self.recording_started_signal.emit()
            self.logger.info(f"Started recording: {filename}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            self.status_signal.emit(f"Failed to start recording: {str(e)}", "error")
            return False
        finally:
            self.mutex.unlock()
    
    def stop_recording(self):
        """Stop video recording"""
        self.mutex.lock()
        try:
            if not self.is_recording:
                self.status_signal.emit("Not currently recording", "warning")
                return False
            
            # Stop recording
            if self.video_recorder:
                filename = self.video_recorder.filename
                self.video_recorder.stop_recording()
                self.video_recorder = None
            
            self.is_recording = False
            
            # Clear remaining frames in queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.status_signal.emit(f"Recording stopped and saved", "info")
            self.recording_stopped_signal.emit(filename if 'filename' in locals() else "")
            self.logger.info("Stopped recording")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop recording: {e}")
            self.status_signal.emit(f"Failed to stop recording: {str(e)}", "error")
            return False
        finally:
            self.mutex.unlock()
    
    def stop(self):
        """Stop the thread"""
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        self.running = False
        self.wait(1000)


class CameraThread(QThread):
    """Camera thread for capturing images from Basler camera"""
    new_image_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.camera = None
        self.frame_rate = 30.0
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        self.running = True
        try:
            # Import here to avoid import errors if not installed
            import pypylon.pylon as pylon
            from pypylon import genicam
            
            # Get the transport layer factory
            tl_factory = pylon.TlFactory.GetInstance()
            
            # Find all available devices
            devices = tl_factory.EnumerateDevices()
            
            if not devices:
                self.error_signal.emit("No cameras found.")
                return
            
            # Create and connect camera
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            logger.info(f"Using device: {self.camera.GetDeviceInfo().GetModelName()}")
            
            # Open camera
            self.camera.Open()
            
            # Configure camera settings
            self._configure_camera()
            
            # Start grabbing
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            logger.info("Started grabbing images")
            
            while self.running and self.camera.IsGrabbing():
                grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                
                if grab_result.GrabSucceeded():
                    # Copy array to avoid data race
                    img_array = grab_result.Array.copy()
                    self.new_image_signal.emit(img_array)
                
                grab_result.Release()
                
        except Exception as e:
            error_msg = f"Camera error: {str(e)}"
            self.logger.error(error_msg)
            self.error_signal.emit(error_msg)
        finally:
            self._cleanup()
    
    def _configure_camera(self):
        """Configure camera parameters"""
        try:
            import pypylon.pylon as pylon
            from pypylon import genicam
            
            # Set to continuous acquisition
            if genicam.IsAvailable(self.camera.TriggerMode):
                self.camera.TriggerMode.SetValue("Off")
            
            # Heartbeat timeout (for GigE cameras)
            if (self.camera.GetDeviceInfo().GetDeviceClass() == "BaslerGigE" and 
                genicam.IsAvailable(self.camera.GevHeartbeatTimeout)):
                self.camera.GevHeartbeatTimeout.SetValue(1000)
            
            # Frame rate settings
            if genicam.IsAvailable(self.camera.AcquisitionFrameRateEnable):
                self.camera.AcquisitionFrameRateEnable.SetValue(True)
                
                if genicam.IsAvailable(self.camera.AcquisitionFrameRateAbs):
                    self.camera.AcquisitionFrameRateAbs.SetValue(self.frame_rate)
                elif genicam.IsAvailable(self.camera.AcquisitionFrameRate):
                    self.camera.AcquisitionFrameRate.SetValue(self.frame_rate)
            
            # Pixel format
            if genicam.IsAvailable(self.camera.PixelFormat):
                self.camera.PixelFormat.SetValue("Mono8")
            
            # ROI settings (1600x1200)
            if (genicam.IsAvailable(self.camera.Width) and 
                genicam.IsAvailable(self.camera.Height)):
                self.camera.OffsetX.SetValue(0)
                self.camera.OffsetY.SetValue(0)
                self.camera.Width.SetValue(1600)
                self.camera.Height.SetValue(1200)
                
        except Exception as e:
            error_msg = f"Failed to configure camera: {str(e)}"
            self.logger.error(error_msg)
            self.error_signal.emit(error_msg)
            raise
    
    def _cleanup(self):
        """Clean up camera resources"""
        try:
            if self.camera:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                    logger.info("Stopped grabbing images")
                
                if self.camera.IsOpen():
                    logger.info(f"Closing camera {self.camera.GetDeviceInfo().GetModelName()}")
                    self.camera.Close()
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def stop(self):
        self.running = False
        self.wait(3000)  # Wait up to 3 seconds for safe shutdown


class DepthDatasetThread(QThread):
    """Thread for collecting Z-depth dataset"""
    status_signal = pyqtSignal(str, str)  # message, type
    collection_progress_signal = pyqtSignal(int, int)  # current, total
    collection_complete_signal = pyqtSignal(bool, str)  # success, message
    roi_marker_signal = pyqtSignal(QPoint)  # ROI center point for display
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.command_queue = queue.Queue()
        self.stage = None
        self.current_image = None
        self.logger = logging.getLogger(__name__)
        
        # Create depth_dataset directory if it doesn't exist
        os.makedirs("depth_dataset", exist_ok=True)
        
        # Z-axis parameters - ensure these are integers for directory naming
        self.z_step = 100  # 50 μm per step (integer)
        self.z_range = 500   # ±500 μm range (integer)
        self.roi_size = 64  # 64x64 ROI (integer)
        
    def set_current_image(self, image):
        """Update current image for ROI extraction"""
        self.current_image = image.copy() if image is not None else None
        
    def run(self):
        self.running = True
        
        try:
            # Initialize stage
            self.stage = Stage()
            self.status_signal.emit("Depth dataset thread initialized", "info")
            
            while self.running:
                try:
                    # Get command from queue with timeout
                    cmd, args = self.command_queue.get(timeout=0.5)
                    
                    if cmd == "collect_dataset":
                        center_x, center_y = args
                        self.collect_z_dataset(center_x, center_y)
                    
                    self.command_queue.task_done()
                    
                except queue.Empty:
                    continue
                    
        except Exception as e:
            error_msg = f"Depth dataset thread error: {str(e)}"
            self.logger.error(error_msg)
            self.status_signal.emit(error_msg, "error")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up stage resources"""
        try:
            if hasattr(self, 'stage') and self.stage:
                self.stage.close()
                self.logger.info("Depth dataset stage closed")
        except Exception as e:
            self.logger.error(f"Depth dataset cleanup error: {e}")
    
    def collect_z_dataset(self, center_x, center_y):
        """Collect Z-depth dataset centered at the given point"""
        try:
            # Emit ROI marker signal for display
            self.roi_marker_signal.emit(QPoint(center_x, center_y))
            
            if self.current_image is None:
                self.collection_complete_signal.emit(False, "No image available for ROI extraction")
                return
            
            # Generate timestamp for image names
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Get initial Z position
            initial_z = self.stage.get_z_position()
            self.status_signal.emit(f"Starting Z-depth collection at ({center_x}, {center_y})", "info")
            
            # Calculate the actual depth values to sample, using z_step
            # Ensure we use integers for calculations
            z_range_int = int(self.z_range)
            z_step_int = int(self.z_step)
            
            num_steps = (2 * z_range_int) // z_step_int + 1
            
            # Generate depth values as integers
            depth_values = []
            for i in range(int(num_steps)):
                depth = -z_range_int + i * z_step_int
                depth_values.append(int(depth))
            
            # Create depth directories
            for depth in depth_values:
                depth_dir = os.path.join("depth_dataset", str(depth))
                os.makedirs(depth_dir, exist_ok=True)
            
            total_images = len(depth_values)
            
            # Collect images at different depths
            for i, depth in enumerate(depth_values):
                # Update progress
                self.collection_progress_signal.emit(i + 1, total_images)
                
                # Move to target Z position
                target_z = initial_z + float(depth)  # Convert to float for stage movement
                self.stage.move_z_to_absolute(target_z)
                
                # Wait for stage to settle
                time.sleep(0.2)
                
                # Extract ROI from current image
                roi = self._extract_roi(center_x, center_y)
                
                if roi is not None:
                    # Save ROI image
                    depth_dir = os.path.join("depth_dataset", str(depth))
                    filename = os.path.join(depth_dir, f"{timestamp}.png")
                    cv2.imwrite(filename, roi)
                    
                    self.status_signal.emit(f"Saved depth {depth} μm: {filename}", "info")
                else:
                    self.logger.warning(f"Failed to extract ROI for depth {depth}")
            
            # Return to initial Z position
            self.stage.move_z_to_absolute(initial_z)
            
            self.collection_complete_signal.emit(True, f"Successfully collected {total_images} depth images")
            
        except Exception as e:
            error_msg = f"Z-depth collection failed: {str(e)}"
            self.logger.error(error_msg)
            self.collection_complete_signal.emit(False, error_msg)
            
            # Try to return to initial position
            try:
                if 'initial_z' in locals():
                    self.stage.move_z_to_absolute(initial_z)
            except:
                pass
    
    def _extract_roi(self, center_x, center_y):
        """Extract 64x64 ROI centered at given coordinates"""
        if self.current_image is None:
            return None
            
        try:
            h, w = self.current_image.shape[:2]
            
            # Calculate ROI bounds
            half_size = self.roi_size // 2
            
            # Ensure ROI is within image bounds
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(w, center_x + half_size)
            y2 = min(h, center_y + half_size)
            
            # Extract ROI
            roi = self.current_image[y1:y2, x1:x2]
            
            # Ensure ROI is exactly 64x64
            if roi.shape[0] != self.roi_size or roi.shape[1] != self.roi_size:
                roi = cv2.resize(roi, (self.roi_size, self.roi_size))
            
            return roi
            
        except Exception as e:
            self.logger.error(f"ROI extraction error: {e}")
            return None
    
    def request_collect_dataset(self, center_x, center_y):
        """Request to collect Z-depth dataset"""
        self.command_queue.put(("collect_dataset", (center_x, center_y)))
    
    def stop(self):
        self.running = False
        self.wait(1000)


class StageThread(QThread):
    """Thread for controlling stage movement and calibration"""
    status_signal = pyqtSignal(str, str)  # message, type (info, warning, error)
    calibration_point_signal = pyqtSignal(int, QPoint)  # Point index, point position
    calibration_complete_signal = pyqtSignal(bool)  # Calibration completed successfully
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.stage = None
        self.command_queue = queue.Queue()
        self.calibration_file = "stage_calibration.txt"
        self.calibration_points = []  # List of image points for calibration
        self.is_calibrated = False
        self.transformation_matrix = None
        self.reference_point = None  # Reference point for movement
        self.logger = logging.getLogger(__name__)
        self.image_width = 1600  # Default values, will be updated
        self.image_height = 1200
        
        # Predefined calibration point positions (relative to image center)
        self.calibration_offsets = [
            (-798, -598),  # Top-left
            (798, -598),   # Top-right
            (798, 598),    # Bottom-right
            (-798, 598),   # Bottom-left
        ]
        
    def set_image_dimensions(self, width, height):
        """Set the current image dimensions for calibration point calculation"""
        self.image_width = width
        self.image_height = height
        
    def run(self):
        self.running = True
        
        try:
            # Initialize stage
            self.stage = Stage()
            self.status_signal.emit("Stage connected successfully", "info")
            
            # Load calibration if exists
            self.load_calibration()
            
            while self.running:
                try:
                    # Get command from queue with timeout
                    cmd, args = self.command_queue.get(timeout=0.5)
                    
                    if cmd == "move_to":
                        x, y = args
                        self.move_to_image_point(x, y)
                    
                    elif cmd == "calibrate":
                        self.start_calibration()
                    
                    elif cmd == "add_calibration_point":
                        img_point, index = args
                        self.add_calibration_point(img_point, index)
                    
                    elif cmd == "set_image_dimensions":
                        width, height = args
                        self.set_image_dimensions(width, height)
                    
                    elif cmd == "set_reference_point":
                        x, y = args
                        self.set_reference_point(x, y)
                    
                    elif cmd == "move_z_relative":
                        z_delta = args
                        self.move_z_relative(z_delta)
                    
                    self.command_queue.task_done()
                    
                except queue.Empty:
                    continue
                    
        except Exception as e:
            error_msg = f"Stage error: {str(e)}"
            self.logger.error(error_msg)
            self.status_signal.emit(error_msg, "error")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up stage resources"""
        try:
            if hasattr(self, 'stage') and self.stage:
                self.stage.close()
                self.logger.info("Stage closed")
        except Exception as e:
            self.logger.error(f"Stage cleanup error: {e}")
    
    def move_to_image_point(self, img_x, img_y):
        """Move stage to position that brings the clicked point to the reference point"""
        if not self.is_calibrated:
            self.status_signal.emit("Stage not calibrated. Please calibrate first.", "warning")
            return
            
        if self.reference_point is None:
            self.status_signal.emit("Reference point not set. Please set a reference point first.", "warning")
            return
        
        try:
            # Get current stage position
            current_stage_pos = self.stage.get_xy_position()
            
            # Calculate the offset between clicked point and reference point in image coordinates
            dx_img = img_x - self.reference_point[0]
            dy_img = img_y - self.reference_point[1]
            
            # Convert image offset to stage offset using the transformation matrix
            dx_stage = dx_img * self.transformation_matrix['x_scale']
            dy_stage = dy_img * self.transformation_matrix['y_scale']
            
            # Calculate the new stage position (move in opposite direction to bring clicked point to reference)
            new_stage_x = current_stage_pos[0] + dx_stage * -1
            new_stage_y = current_stage_pos[1] + dy_stage
            
            # Move the stage
            self.stage.move_xy_to_absolute(new_stage_x, new_stage_y)
            
            self.status_signal.emit(f"Moving to position: ({new_stage_x:.2f}, {new_stage_y:.2f})", "info")
            
        except Exception as e:
            self.logger.error(f"Move error: {e}")
            self.status_signal.emit(f"Move failed: {str(e)}", "error")
    
    def move_z_relative(self, z_delta):
        """Move Z stage by relative amount"""
        try:
            # Get current Z position for logging
            current_z = self.stage.get_z_position()
            
            # Move Z stage
            self.stage.move_z_relative(z_delta)
            
            # Get new Z position
            new_z = self.stage.get_z_position()
            
            self.status_signal.emit(f"Z moved from {current_z:.2f} to {new_z:.2f} (Δz: {z_delta}) μm", "info")
            
        except Exception as e:
            self.logger.error(f"Z move error: {e}")
            self.status_signal.emit(f"Z move failed: {str(e)}", "error")
    
    def set_reference_point(self, x, y):
        """Set the reference point for stage movement"""
        if not self.is_calibrated:
            self.status_signal.emit("Cannot set reference point: Stage not calibrated. Please calibrate first.", "warning")
            return False
        
        self.reference_point = (x, y)
        self.status_signal.emit(f"Reference point set to ({x}, {y})", "info")
        self.save_calibration()  # Save the updated reference point
        return True
    
    def start_calibration(self):
        """Start the calibration process"""
        self.calibration_points = []
        self.is_calibrated = False
        self.reference_point = None  # Clear reference point
        
        # Request first calibration point
        self.request_next_calibration_point()
    
    def request_next_calibration_point(self):
        """Request the next calibration point"""
        point_index = len(self.calibration_points)
        
        if point_index >= len(self.calibration_offsets):
            # All points collected, complete calibration
            self.complete_calibration()
            return
        
        # Calculate the next point position based on image center and offset
        img_center_x = self.image_width // 2
        img_center_y = self.image_height // 2
        
        # Get the offset for this calibration point
        offset_x, offset_y = self.calibration_offsets[point_index]
        
        # Calculate the point position
        point_x = img_center_x + offset_x
        point_y = img_center_y + offset_y
        
        # Send signal to display the point
        self.calibration_point_signal.emit(point_index, QPoint(point_x, point_y))
        
        # Update status
        point_names = ["top-left", "top-right", "bottom-right", "bottom-left"]
        self.status_signal.emit(f"Please click on the {point_names[point_index]} calibration point (red marker)", "warning")
    
    def add_calibration_point(self, img_point, index):
        """Add a calibration point"""
        self.calibration_points.append((img_point.x(), img_point.y()))
        
        self.logger.info(f"Added calibration point {index+1}: Image({img_point.x()}, {img_point.y()})")
        
        # Request next calibration point
        self.request_next_calibration_point()
    
    def complete_calibration(self):
        """Complete the calibration process"""
        if len(self.calibration_points) < 4:
            self.status_signal.emit("Calibration failed: Need at least 4 points.", "error")
            self.calibration_complete_signal.emit(False)
            return
            
        try:
            # Calculate transformation matrix
            self.calculate_transformation()
            
            # Set default reference point (center of calibration points)
            x_sum = sum(p[0] for p in self.calibration_points)
            y_sum = sum(p[1] for p in self.calibration_points)
            self.reference_point = (x_sum / len(self.calibration_points), y_sum / len(self.calibration_points))
            
            # Save calibration
            self.save_calibration()
            
            self.is_calibrated = True
            self.status_signal.emit("Calibration completed successfully. Please set a reference point.", "info")
            self.calibration_complete_signal.emit(True)
            
        except Exception as e:
            self.logger.error(f"Calibration error: {e}")
            self.status_signal.emit(f"Calibration failed: {str(e)}", "error")
            self.calibration_complete_signal.emit(False)
    
    def calculate_transformation(self):
        """Calculate the transformation matrix from image to stage coordinates"""
        # Calculate horizontal and vertical distances between points
        # Points are arranged in a rectangle: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left
        
        # Horizontal distances (points 0-1 and 3-2)
        dx_img_top = abs(self.calibration_points[1][0] - self.calibration_points[0][0])
        dx_img_bottom = abs(self.calibration_points[2][0] - self.calibration_points[3][0])
        
        # Vertical distances (points 0-3 and 1-2)
        dy_img_left = abs(self.calibration_points[3][1] - self.calibration_points[0][1])
        dy_img_right = abs(self.calibration_points[2][1] - self.calibration_points[1][1])
        
        # Known physical distances in um (20x objective)
        dx_physical = 370.0   # um
        dy_physical = 270.0   # um
        
        # Calculate scale factors (um per pixel)
        x_scales = [dx_physical / dx_img_top, dx_physical / dx_img_bottom]
        y_scales = [dy_physical / dy_img_left, dy_physical / dy_img_right]
        
        # Average the scale factors
        x_scale_avg = sum(x_scales) / len(x_scales)
        y_scale_avg = sum(y_scales) / len(y_scales)
        
        # Store the transformation matrix
        self.transformation_matrix = {
            'x_scale': x_scale_avg,
            'y_scale': y_scale_avg
        }
        
        self.logger.info(f"Transformation matrix: x_scale={x_scale_avg:.6f}, y_scale={y_scale_avg:.6f}")
    
    def save_calibration(self):
        """Save calibration data to file"""
        try:
            with open(self.calibration_file, 'w') as f:
                # Save transformation matrix
                f.write(f"{self.transformation_matrix['x_scale']},{self.transformation_matrix['y_scale']}\n")
                
                # Save reference point (if set)
                if self.reference_point:
                    f.write(f"{self.reference_point[0]},{self.reference_point[1]}\n")
                else:
                    f.write("None,None\n")
                
                # Save calibration points
                for point in self.calibration_points:
                    f.write(f"{point[0]},{point[1]}\n")
                    
            self.logger.info(f"Calibration saved to {self.calibration_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save calibration: {e}")
    
    def load_calibration(self):
        """Load calibration data from file"""
        if not os.path.exists(self.calibration_file):
            self.logger.info("No calibration file found.")
            return
            
        try:
            with open(self.calibration_file, 'r') as f:
                lines = f.readlines()
                
                if len(lines) < 6:  # Need at least transformation matrix, reference point, and 4 calibration points
                    self.logger.error("Invalid calibration file format.")
                    return
                
                # Load transformation matrix
                x_scale, y_scale = map(float, lines[0].strip().split(','))
                self.transformation_matrix = {
                    'x_scale': x_scale,
                    'y_scale': y_scale
                }
                
                # Load reference point
                ref_x_str, ref_y_str = lines[1].strip().split(',')
                if ref_x_str != "None" and ref_y_str != "None":
                    ref_x, ref_y = float(ref_x_str), float(ref_y_str)
                    self.reference_point = (ref_x, ref_y)
                
                # Load calibration points
                self.calibration_points = []
                for i in range(2, len(lines)):
                    if lines[i].strip():
                        x, y = map(float, lines[i].strip().split(','))
                        self.calibration_points.append((x, y))
                
                self.is_calibrated = True
                self.logger.info("Calibration loaded successfully.")
                status_msg = "Calibration loaded from file."
                if self.reference_point is None:
                    status_msg += " Reference point not set. Please set a reference point."
                self.status_signal.emit(status_msg, "info")
                
        except Exception as e:
            self.logger.error(f"Failed to load calibration: {e}")
    
    # Request methods for external calls
    def request_move_to(self, img_x, img_y):
        """Request to move the stage to the given image coordinates"""
        self.command_queue.put(("move_to", (img_x, img_y)))
    
    def request_move_z_relative(self, z_delta):
        """Request to move the Z stage by a relative amount"""
        self.command_queue.put(("move_z_relative", z_delta))
    
    def request_calibration(self):
        """Request to start calibration"""
        self.command_queue.put(("calibrate", None))
    
    def request_add_calibration_point(self, img_point, index):
        """Request to add a calibration point"""
        self.command_queue.put(("add_calibration_point", (img_point, index)))
    
    def request_set_image_dimensions(self, width, height):
        """Request to set image dimensions"""
        self.command_queue.put(("set_image_dimensions", (width, height)))
    
    def request_set_reference_point(self, x, y):
        """Request to set reference point"""
        self.command_queue.put(("set_reference_point", (x, y)))
    
    def stop(self):
        self.running = False
        self.wait(1000)


class ArmThread(QThread):
    """Thread for controlling robotic arm movement"""
    status_signal = pyqtSignal(str, str)  # message, type
    position_signal = pyqtSignal(int, int, int)  # x_steps, y_steps, z_steps
    connection_status_signal = pyqtSignal(bool, bool, bool, bool)  # success, x_state, y_state, z_state
    
    def __init__(self, arm_id=1):
        super().__init__()
        self.running = False
        self.command_queue = queue.Queue()
        self.arm_id = arm_id  # Arm ID for identification
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        self.running = True
        
        while self.running:
            try:
                # Get command from queue with timeout
                cmd, args = self.command_queue.get(timeout=0.5)
                
                if cmd == "check_connection":
                    self.check_connection_status()
                
                elif cmd == "check_working":
                    self.check_working_status()
                
                elif cmd == "get_position":
                    self.get_motor_position()
                
                elif cmd == "move":
                    motor_type, speed, step = args
                    self.move_motor(motor_type, speed, step)
                
                self.command_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Arm thread error: {e}")
                self.status_signal.emit(f"Arm error: {str(e)}", "error")
    
    def check_connection_status(self):
        """Check arm connection status"""
        try:
            success, x_state, y_state, z_state = armConnectState(self.arm_id)
            self.connection_status_signal.emit(success, x_state, y_state, z_state)
            
            if success:
                self.status_signal.emit(f"Arm {self.arm_id} connection status: X:{x_state}, Y:{y_state}, Z:{z_state}", "info")
            else:
                self.status_signal.emit(f"Failed to get Arm {self.arm_id} connection status", "error")
                
        except Exception as e:
            self.logger.error(f"Connection check error: {e}")
            self.status_signal.emit(f"Connection check error: {str(e)}", "error")
    
    def check_working_status(self):
        """Check arm working status"""
        try:
            success, x_state, y_state, z_state = armWorkingState(self.arm_id)
            
            if success:
                self.status_signal.emit(f"Arm {self.arm_id} working status: X:{x_state}, Y:{y_state}, Z:{z_state}", "info")
            else:
                self.status_signal.emit(f"Failed to get Arm {self.arm_id} working status", "error")
                
        except Exception as e:
            self.logger.error(f"Working status check error: {e}")
            self.status_signal.emit(f"Working status check error: {str(e)}", "error")
    
    def get_motor_position(self):
        """Get arm motor position"""
        try:
            success, x_steps, y_steps, z_steps = armGetMotorPos(self.arm_id)
            
            if success:
                self.position_signal.emit(x_steps, y_steps, z_steps)
                self.status_signal.emit(f"Arm {self.arm_id} position: X:{x_steps}, Y:{y_steps}, Z:{z_steps} μm", "info")
            else:
                self.status_signal.emit(f"Failed to get Arm {self.arm_id} position", "error")
                
        except Exception as e:
            self.logger.error(f"Position get error: {e}")
            self.status_signal.emit(f"Position get error: {str(e)}", "error")
    
    def move_motor(self, motor_type, speed, step):
        """Move arm motor"""
        try:
            success, x, y, z = armGetMotorPos(self.arm_id)
            if success:
                self.current_x = x
                self.current_y = y
                self.current_z = z
            target_pos = 0
            if motor_type == 1:
                target_pos = x - step
            elif motor_type == 2:
                target_pos = y + step
            elif motor_type == 3:
                target_pos = z + step
            success, result = armMovebyPos(self.arm_id, motor_type, speed, target_pos)
            
            if success:
                motor_names = {1: "X", 2: "Y", 3: "Z"}
                motor_name = motor_names.get(motor_type, "Unknown")
                self.status_signal.emit(f"Arm {self.arm_id} {motor_name} motor moved by {step} steps at speed {speed}", "info")
            else:
                self.status_signal.emit(f"Failed to move Arm {self.arm_id} motor {motor_type}", "error")
                
        except Exception as e:
            self.logger.error(f"Motor move error: {e}")
            self.status_signal.emit(f"Motor move error: {str(e)}", "error")
    
    # Request methods for external calls
    def request_check_connection(self):
        """Request to check connection status"""
        self.command_queue.put(("check_connection", None))
    
    def request_check_working(self):
        """Request to check working status"""
        self.command_queue.put(("check_working", None))
    
    def request_get_position(self):
        """Request to get motor position"""
        self.command_queue.put(("get_position", None))
    
    def request_move_motor(self, motor_type, speed, step):
        """Request to move motor"""
        self.command_queue.put(("move", (motor_type, speed, step)))
    
    def stop(self):
        self.running = False
        self.wait(1000)


class PumpControlThread(QThread):
    """Thread for controlling pump operations"""
    status_signal = pyqtSignal(str, str)  # message, type
    position_signal = pyqtSignal(float, float, float)  # position data from pump
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.command_queue = queue.Queue()
        self.pump_thread = None
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        self.running = True
        
        try:
            # Initialize pump thread
            self.pump_thread = LeftPumpThread()
            self.pump_thread.position_updated.connect(self.position_signal)
            self.pump_thread.start()
            self.status_signal.emit("Pump connected successfully", "info")
            
            while self.running:
                try:
                    # Get command from queue with timeout
                    cmd, args = self.command_queue.get(timeout=0.5)
                    
                    if cmd == "step_cw":
                        step = args
                        self.step_clockwise(step)
                    
                    elif cmd == "step_ccw":
                        step = args
                        self.step_counter_clockwise(step)
                    
                    elif cmd == "set_speed":
                        speed = args
                        self.set_speed(speed)
                    
                    elif cmd == "run":
                        speed = args
                        self.run_continuous(speed)
                    
                    self.command_queue.task_done()
                    
                except queue.Empty:
                    continue
                    
        except Exception as e:
            error_msg = f"Pump error: {str(e)}"
            self.logger.error(error_msg)
            self.status_signal.emit(error_msg, "error")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up pump resources"""
        try:
            if self.pump_thread:
                self.pump_thread.shutdown()
                self.pump_thread.wait(1000)
                self.logger.info("Pump closed")
        except Exception as e:
            self.logger.error(f"Pump cleanup error: {e}")
    
    def step_clockwise(self, step):
        """Move pump clockwise by specified steps"""
        try:
            if self.pump_thread:
                self.pump_thread.add_task(("Step_CW", step))
                self.status_signal.emit(f"Pump stepping clockwise by {step} steps", "info")
        except Exception as e:
            self.logger.error(f"Step CW error: {e}")
            self.status_signal.emit(f"Step CW failed: {str(e)}", "error")
    
    def step_counter_clockwise(self, step):
        """Move pump counter-clockwise by specified steps"""
        try:
            if self.pump_thread:
                self.pump_thread.add_task(("Step_CCW", step))
                self.status_signal.emit(f"Pump stepping counter-clockwise by {step} steps", "info")
        except Exception as e:
            self.logger.error(f"Step CCW error: {e}")
            self.status_signal.emit(f"Step CCW failed: {str(e)}", "error")
    
    def set_speed(self, speed):
        """Set pump speed"""
        try:
            if self.pump_thread:
                self.pump_thread.add_task(("SetSpeed", speed))
                self.status_signal.emit(f"Pump speed set to {speed}", "info")
        except Exception as e:
            self.logger.error(f"Set speed error: {e}")
            self.status_signal.emit(f"Set speed failed: {str(e)}", "error")
    
    def run_continuous(self, speed):
        """Run pump continuously at specified speed"""
        try:
            if self.pump_thread:
                self.pump_thread.add_task(("Run", speed))
                self.status_signal.emit(f"Pump running continuously at speed {speed}", "info")
        except Exception as e:
            self.logger.error(f"Run continuous error: {e}")
            self.status_signal.emit(f"Run continuous failed: {str(e)}", "error")
    
    # Request methods for external calls
    def request_step_cw(self, step):
        """Request clockwise step"""
        self.command_queue.put(("step_cw", step))
    
    def request_step_ccw(self, step):
        """Request counter-clockwise step"""
        self.command_queue.put(("step_ccw", step))
    
    def request_set_speed(self, speed):
        """Request speed setting"""
        self.command_queue.put(("set_speed", speed))
    
    def request_run(self, speed):
        """Request continuous running"""
        self.command_queue.put(("run", speed))
    
    def stop(self):
        self.running = False
        self.wait(1000)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscope Control System - Sperm Selection")
        
        # Set up the UI
        self._setup_ui()
        
        # Initialize threads
        self._setup_threads()

        # YOLO detection state
        self.yolo_detecting = False
        self.detection_overlay = None  # Stores detection results for overlay
        
        # Calibration state
        self.calibration_mode = False
        self.calibration_point_index = -1
        self.current_calibration_point = None
        
        # Reference point setting mode
        self.reference_point_mode = False
        
        # Depth dataset collection mode
        self.depth_collection_mode = False
        
        # Visual feedback markers
        self.target_marker = None
        self.reference_marker = None
        self.roi_marker = None
        self.marker_timer = QTimer()
        self.marker_timer.timeout.connect(self.clear_markers)
        
        # Z-axis movement step size
        self.z_step = 50.0  # 5 μm per scroll step
        
        # Recording state
        self.is_recording = False
        self.current_image_size = (1600, 1200)  # Default size
        
        # Current image for ROI extraction
        self.current_image = None
        
        # Start threads
        self.start_threads()
        
        # Set focus to image label to receive key events
        self.image_label.setFocus()
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)  # Horizontal layout
        
        # Left side: Image display
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.main_layout.addWidget(self.left_panel, 1)  # Give it a stretch factor of 1
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.left_layout.addWidget(self.image_label)
        
        # Status label
        self.status_label = QLabel("Starting... Double-click on image to move stage (requires calibration)")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("background-color: #FFFFCC; color: #CC6600; padding: 5px;")
        self.status_label.setFixedHeight(30)
        self.left_layout.addWidget(self.status_label)
        
        # Right side: Control panel
        self.right_panel = QWidget()
        self.right_panel.setFixedWidth(350)  # Increased width for more controls
        self.right_layout = QVBoxLayout(self.right_panel)
        self.main_layout.addWidget(self.right_panel, 0)  # Don't stretch
        
        # Setup control groups
        self._setup_video_recording_group()
        self._setup_detection_group()
        self._setup_stage_control_group()
        self._setup_arm_control_group()
        self._setup_pump_control_group()
        
        # Add groups to control panel
        self.right_layout.addWidget(self.recording_group)
        self.right_layout.addSpacing(10)
        self.right_layout.addWidget(self.detection_group)  # Add this line
        self.right_layout.addSpacing(10)
        self.right_layout.addWidget(self.stage_group)
        self.right_layout.addSpacing(10)
        self.right_layout.addWidget(self.arm_group)
        self.right_layout.addSpacing(10)
        self.right_layout.addWidget(self.pump_group)
        self.right_layout.addStretch(1)
        
        # Connect mouse events for the image label
        self.image_label.mousePressEvent = self.image_click
        self.image_label.mouseDoubleClickEvent = self.image_double_click
        self.image_label.wheelEvent = self.image_wheel_event
    
    def _setup_video_recording_group(self):
        """Setup video recording control group"""
        self.recording_group = QGroupBox("Video Recording")
        recording_layout = QVBoxLayout()
        self.recording_group.setLayout(recording_layout)
        
        # Recording buttons
        recording_btn_layout = QHBoxLayout()
        
        self.start_recording_btn = QPushButton("Start Recording")
        self.start_recording_btn.setFixedHeight(40)
        self.start_recording_btn.setStyleSheet("background-color: #99FF99;")
        self.start_recording_btn.clicked.connect(self.start_recording)
        recording_btn_layout.addWidget(self.start_recording_btn)
        
        self.stop_recording_btn = QPushButton("Stop Recording")
        self.stop_recording_btn.setFixedHeight(40)
        self.stop_recording_btn.setStyleSheet("background-color: #FF9999;")
        self.stop_recording_btn.setEnabled(False)
        self.stop_recording_btn.clicked.connect(self.stop_recording)
        recording_btn_layout.addWidget(self.stop_recording_btn)
        
        recording_layout.addLayout(recording_btn_layout)
        
        # Recording status label
        self.recording_status_label = QLabel("Not recording")
        self.recording_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recording_status_label.setStyleSheet("background-color: #F0F0F0; padding: 5px;")
        recording_layout.addWidget(self.recording_status_label)
    
    def _setup_detection_group(self):
        """Setup YOLO detection control group"""
        self.detection_group = QGroupBox("YOLO Detection & Tracking")
        detection_layout = QVBoxLayout()
        self.detection_group.setLayout(detection_layout)
        
        # Detection toggle button
        self.detect_track_btn = QPushButton("Start Detect & Track")
        self.detect_track_btn.setFixedHeight(50)
        self.detect_track_btn.setStyleSheet("background-color: #99CCFF; font-weight: bold; font-size: 14px;")
        self.detect_track_btn.clicked.connect(self.toggle_detection)
        detection_layout.addWidget(self.detect_track_btn)
        
        # FPS display
        fps_layout = QHBoxLayout()
        
        self.detection_fps_label = QLabel("Detection FPS: --")
        self.detection_fps_label.setStyleSheet("background-color: #F0F0F0; padding: 5px;")
        fps_layout.addWidget(self.detection_fps_label)
        
        self.tracking_fps_label = QLabel("Tracking FPS: --")
        self.tracking_fps_label.setStyleSheet("background-color: #F0F0F0; padding: 5px;")
        fps_layout.addWidget(self.tracking_fps_label)
        
        detection_layout.addLayout(fps_layout)
        
        # Detection status
        self.detection_status_label = QLabel("Detection: OFF")
        self.detection_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detection_status_label.setStyleSheet("background-color: #FFE6E6; color: #CC0000; padding: 5px;")
        detection_layout.addWidget(self.detection_status_label)
        
        # Frame skip control
        skip_layout = QFormLayout()
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(0, 10)
        self.frame_skip_spin.setValue(0)
        self.frame_skip_spin.setToolTip("0 = process every frame, 1 = skip 1 frame, etc.")
        self.frame_skip_spin.valueChanged.connect(self.update_frame_skip)
        skip_layout.addRow("Frame Skip:", self.frame_skip_spin)
        detection_layout.addLayout(skip_layout)

    def _setup_stage_control_group(self):
        """Setup stage control group"""
        self.stage_group = QGroupBox("Stage Control")
        stage_layout = QVBoxLayout()
        self.stage_group.setLayout(stage_layout)
        
        # Calibration button
        self.calibrate_btn = QPushButton("Calibrate Stage")
        self.calibrate_btn.setFixedHeight(40)
        self.calibrate_btn.clicked.connect(self.start_calibration)
        stage_layout.addWidget(self.calibrate_btn)
        
        # Set Reference Point button
        self.set_ref_point_btn = QPushButton("Set Reference Point")
        self.set_ref_point_btn.setFixedHeight(40)
        self.set_ref_point_btn.clicked.connect(self.start_set_reference_point)
        stage_layout.addWidget(self.set_ref_point_btn)
        
        # Collect Z Dataset button
        self.collect_z_dataset_btn = QPushButton("Collect Z Dataset")
        self.collect_z_dataset_btn.setFixedHeight(40)
        self.collect_z_dataset_btn.setStyleSheet("background-color: #FFCC99; font-weight: bold;")
        self.collect_z_dataset_btn.clicked.connect(self.start_collect_z_dataset)
        stage_layout.addWidget(self.collect_z_dataset_btn)
        
        # Z-axis control info
        z_info_label = QLabel("Use mouse wheel to control Z-axis: Scroll up to rise, down to fall.")
        z_info_label.setStyleSheet("color: #0066CC; font-style: italic;")
        z_info_label.setWordWrap(True)
        stage_layout.addWidget(z_info_label)
        
        # Dataset collection progress
        self.dataset_progress_label = QLabel("")
        self.dataset_progress_label.setStyleSheet("background-color: #E6F3FF; color: #0066CC; padding: 5px;")
        self.dataset_progress_label.setWordWrap(True)
        self.dataset_progress_label.hide()
        stage_layout.addWidget(self.dataset_progress_label)
    
    def _setup_arm_control_group(self):
        """Setup arm control group"""
        self.arm_group = QGroupBox("Robotic Arm Control")
        arm_layout = QVBoxLayout()
        self.arm_group.setLayout(arm_layout)
        
        # Arm connection status
        self.arm_connect_btn = QPushButton("Check Arm Connection")
        self.arm_connect_btn.clicked.connect(self.check_arm_connection)
        arm_layout.addWidget(self.arm_connect_btn)
        
        # Movement distance setting
        distance_layout = QFormLayout()
        self.arm_distance_spin = QSpinBox()
        self.arm_distance_spin.setRange(-10000, 10000)
        self.arm_distance_spin.setValue(100)
        self.arm_distance_spin.setSuffix(" μm")
        distance_layout.addRow("Move Distance:", self.arm_distance_spin)
        
        # Movement speed setting
        self.arm_speed_spin = QSpinBox()
        self.arm_speed_spin.setRange(1, 10000)
        self.arm_speed_spin.setValue(100)
        self.arm_speed_spin.setSuffix(" μm/s")
        distance_layout.addRow("Move Speed:", self.arm_speed_spin)
        
        arm_layout.addLayout(distance_layout)
        
        # Movement buttons
        move_grid = QGridLayout()
        
        # Up button (Y+ direction)
        self.arm_up_btn = QPushButton("↑ Up")
        self.arm_up_btn.clicked.connect(lambda: self.move_arm(2, self.arm_speed_spin.value(), self.arm_distance_spin.value()))
        move_grid.addWidget(self.arm_up_btn, 0, 1)
        
        # Left and Right buttons (X directions)
        self.arm_left_btn = QPushButton("← Left")
        self.arm_left_btn.clicked.connect(lambda: self.move_arm(1, self.arm_speed_spin.value(), -self.arm_distance_spin.value()))
        move_grid.addWidget(self.arm_left_btn, 1, 0)
        
        self.arm_right_btn = QPushButton("Right →")
        self.arm_right_btn.clicked.connect(lambda: self.move_arm(1, self.arm_speed_spin.value(), self.arm_distance_spin.value()))
        move_grid.addWidget(self.arm_right_btn, 1, 2)
        
        # Down button (Y- direction)
        self.arm_down_btn = QPushButton("↓ Down")
        self.arm_down_btn.clicked.connect(lambda: self.move_arm(2, self.arm_speed_spin.value(), -self.arm_distance_spin.value()))
        move_grid.addWidget(self.arm_down_btn, 2, 1)
        
        # Z axis buttons
        self.arm_z_up_btn = QPushButton("Z ↑")
        self.arm_z_up_btn.clicked.connect(lambda: self.move_arm(3, self.arm_speed_spin.value(), self.arm_distance_spin.value()))
        move_grid.addWidget(self.arm_z_up_btn, 0, 0)
        
        self.arm_z_down_btn = QPushButton("Z ↓")
        self.arm_z_down_btn.clicked.connect(lambda: self.move_arm(3, self.arm_speed_spin.value(), -self.arm_distance_spin.value()))
        move_grid.addWidget(self.arm_z_down_btn, 2, 0)
        
        # Center position display
        self.arm_pos_btn = QPushButton("Get Position")
        self.arm_pos_btn.clicked.connect(self.get_arm_position)
        move_grid.addWidget(self.arm_pos_btn, 1, 1)
        
        arm_layout.addLayout(move_grid)
        
        # Arm status label
        self.arm_status_label = QLabel("Arm status: Not connected")
        self.arm_status_label.setStyleSheet("background-color: #FFE6CC; color: #FF6600; padding: 5px;")
        self.arm_status_label.setWordWrap(True)
        arm_layout.addWidget(self.arm_status_label)
    
    def _setup_pump_control_group(self):
        """Setup pump control group"""
        self.pump_group = QGroupBox("Pump Control")
        pump_layout = QVBoxLayout()
        self.pump_group.setLayout(pump_layout)
        
        # Pump step setting
        step_layout = QFormLayout()
        self.pump_step_spin = QSpinBox()
        self.pump_step_spin.setRange(1, 50000)
        self.pump_step_spin.setValue(1000)
        self.pump_step_spin.setSuffix(" steps")
        step_layout.addRow("Step Size:", self.pump_step_spin)
        
        # Pump speed setting
        self.pump_speed_spin = QSpinBox()
        self.pump_speed_spin.setRange(100, 40000)
        self.pump_speed_spin.setValue(1000)
        self.pump_speed_spin.setSuffix(" RPM")
        step_layout.addRow("Speed:", self.pump_speed_spin)
        
        pump_layout.addLayout(step_layout)
        
        # Pump control buttons
        pump_btn_layout = QHBoxLayout()

        # Counter-clockwise (negative) rotation button
        self.pump_ccw_btn = QPushButton("Step CCW (Inject)")
        self.pump_ccw_btn.setStyleSheet("background-color: #FF9999;")
        self.pump_ccw_btn.clicked.connect(self.pump_step_ccw)
        pump_btn_layout.addWidget(self.pump_ccw_btn)
        
        # Clockwise (positive) rotation button
        self.pump_cw_btn = QPushButton("Step CW (Withdraw)")
        self.pump_cw_btn.setStyleSheet("background-color: #99FF99;")
        self.pump_cw_btn.clicked.connect(self.pump_step_cw)
        pump_btn_layout.addWidget(self.pump_cw_btn)
        
        pump_layout.addLayout(pump_btn_layout)
        
        # Additional pump controls
        pump_control_layout = QHBoxLayout()
        
        self.pump_set_speed_btn = QPushButton("Set Speed")
        self.pump_set_speed_btn.clicked.connect(self.pump_set_speed)
        pump_control_layout.addWidget(self.pump_set_speed_btn)
        
        self.pump_run_btn = QPushButton("Run Continuous")
        self.pump_run_btn.clicked.connect(self.pump_run)
        pump_control_layout.addWidget(self.pump_run_btn)
        
        pump_layout.addLayout(pump_control_layout)
        
        # Pump status label
        self.pump_status_label = QLabel("Pump status: Initializing...")
        self.pump_status_label.setStyleSheet("background-color: #E6F3FF; color: #0066CC; padding: 5px;")
        self.pump_status_label.setWordWrap(True)
        pump_layout.addWidget(self.pump_status_label)
    
    def _setup_threads(self):
        """Initialize all control threads"""
        # Create threads
        self.camera_thread = CameraThread()
        self.stage_thread = StageThread()
        self.arm_thread = ArmThread(arm_id=1)  # Use arm ID 1
        self.pump_thread = PumpControlThread()
        self.video_recording_thread = VideoRecordingThread()
        self.depth_dataset_thread = DepthDatasetThread()
        # Create YOLO detection thread
        self.yolo_thread = YOLODetectionThread()
        
        # Connect camera signals
        self.camera_thread.new_image_signal.connect(self.update_display)
        self.camera_thread.new_image_signal.connect(self.video_recording_thread.update_frame)
        self.camera_thread.new_image_signal.connect(self.update_current_image)
        self.camera_thread.error_signal.connect(self.show_error)

        # Connect YOLO signals
        self.yolo_thread.detection_result_signal.connect(self.update_detection_overlay)
        self.yolo_thread.status_signal.connect(self.update_detection_status)
        self.yolo_thread.fps_signal.connect(self.update_fps_display)
        
        # Connect stage signals
        self.stage_thread.status_signal.connect(self.update_status)
        self.stage_thread.calibration_point_signal.connect(self.show_calibration_point)
        self.stage_thread.calibration_complete_signal.connect(self.on_calibration_complete)
        
        # Connect arm signals
        self.arm_thread.status_signal.connect(self.update_arm_status)
        self.arm_thread.position_signal.connect(self.update_arm_position)
        self.arm_thread.connection_status_signal.connect(self.update_arm_connection_status)
        
        # Connect pump signals
        self.pump_thread.status_signal.connect(self.update_pump_status)
        self.pump_thread.position_signal.connect(self.update_pump_position)
        
        # Connect video recording signals
        self.video_recording_thread.status_signal.connect(self.update_recording_status)
        self.video_recording_thread.recording_started_signal.connect(self.on_recording_started)
        self.video_recording_thread.recording_stopped_signal.connect(self.on_recording_stopped)
        
        # Connect depth dataset signals
        self.depth_dataset_thread.status_signal.connect(self.update_status)
        self.depth_dataset_thread.collection_progress_signal.connect(self.update_dataset_progress)
        self.depth_dataset_thread.collection_complete_signal.connect(self.on_dataset_collection_complete)
        self.depth_dataset_thread.roi_marker_signal.connect(self.show_roi_marker)
    
    def start_threads(self):
        """Start all control threads"""
        self.stage_thread.start()
        self.camera_thread.start()
        self.arm_thread.start()
        self.pump_thread.start()
        self.video_recording_thread.start()
        self.depth_dataset_thread.start()
        self.yolo_thread.start()
    
    def update_current_image(self, image):
        """Update current image for depth dataset thread"""
        self.current_image = image.copy() if image is not None else None
        # Update YOLO thread with new frame if detecting
        if self.yolo_detecting and self.yolo_thread:
            self.yolo_thread.update_frame(image)
        if self.depth_dataset_thread:
            self.depth_dataset_thread.set_current_image(self.current_image)
    
    def update_display(self, image):
        """Update the display with camera image"""
        try:
            h, w = image.shape[:2]
            self.current_image_size = (w, h)
            
            # Update stage thread with current image dimensions
            self.stage_thread.request_set_image_dimensions(w, h)
            
            # Convert grayscale to RGB for display
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            bytes_per_line = 3 * w
            q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Create pixmap from QImage
            pixmap = QPixmap.fromImage(q_image)
            
            # Draw markers if needed
            painter = QPainter(pixmap)
            
            # Draw recording indicator
            if self.is_recording:
                painter.setPen(QPen(QColor(255, 0, 0), 2))
                painter.setBrush(QColor(255, 0, 0))
                painter.drawEllipse(20, 20, 20, 20)
                
                font = QFont()
                font.setPointSize(14)
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QColor(255, 0, 0))
                painter.drawText(50, 35, "REC")
            
            # Draw calibration point (red)
            if self.calibration_mode and self.current_calibration_point:
                painter.setPen(QPen(QColor(255, 0, 0), 3))
                painter.drawEllipse(self.current_calibration_point, 10, 10)
                painter.drawLine(self.current_calibration_point.x() - 15, self.current_calibration_point.y(),
                               self.current_calibration_point.x() + 15, self.current_calibration_point.y())
                painter.drawLine(self.current_calibration_point.x(), self.current_calibration_point.y() - 15,
                               self.current_calibration_point.x(), self.current_calibration_point.y() + 15)
            
            # Draw reference point (green)
            if self.reference_marker:
                painter.setPen(QPen(QColor(0, 255, 0), 3))
                painter.drawEllipse(self.reference_marker, 12, 12)
                painter.drawLine(self.reference_marker.x() - 18, self.reference_marker.y(),
                               self.reference_marker.x() + 18, self.reference_marker.y())
                painter.drawLine(self.reference_marker.x(), self.reference_marker.y() - 18,
                               self.reference_marker.x(), self.reference_marker.y() + 18)
            
            # Draw target marker (blue)
            if self.target_marker:
                painter.setPen(QPen(QColor(0, 120, 255), 3))
                painter.drawEllipse(self.target_marker, 10, 10)
                painter.drawLine(self.target_marker.x() - 15, self.target_marker.y(),
                               self.target_marker.x() + 15, self.target_marker.y())
                painter.drawLine(self.target_marker.x(), self.target_marker.y() - 15,
                               self.target_marker.x(), self.target_marker.y() + 15)
            
            # Draw detection overlay if available
            if self.detection_overlay is not None and self.yolo_detecting:
                boxes, scores, class_ids, tracks = self.detection_overlay
                
                # Draw bounding boxes
                for box, score, class_id in zip(boxes, scores, class_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Get class color
                    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                    color = base_colors[int(class_id) % len(base_colors)]
                    
                    # Draw box
                    painter.setPen(QPen(QColor(*color), 2))
                    painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                    
                    # Draw label
                    label = f"{self.yolo_thread.class_names[int(class_id)]}: {score:.2f}"
                    font = QFont()
                    font.setPointSize(10)
                    painter.setFont(font)
                    
                    # Draw background for text
                    text_rect = painter.fontMetrics().boundingRect(label)
                    painter.fillRect(x1, y1 - text_rect.height() - 2, text_rect.width() + 4, text_rect.height() + 2, QColor(*color))
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(x1 + 2, y1 - 2, label)
                
                # Draw tracks
                for track_id, track_data in tracks.items():
                    trajectory = track_data['trajectory']
                    color = track_data['color']
                    
                    if len(trajectory) >= 2:
                        painter.setPen(QPen(QColor(*color), 2))
                        for i in range(1, len(trajectory)):
                            pt1 = trajectory[i-1]
                            pt2 = trajectory[i]
                            painter.drawLine(int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))
                        
                        # Draw track ID
                        last_pt = trajectory[-1]
                        painter.setPen(QColor(*color))
                        painter.drawText(int(last_pt[0]), int(last_pt[1]) - 10, f"ID:{track_id}")

            # Draw ROI marker (orange) for depth dataset collection
            if self.roi_marker:
                painter.setPen(QPen(QColor(255, 165, 0), 4))
                # Draw circle
                painter.drawEllipse(self.roi_marker, 32, 32)  # 64x64 ROI, so radius is 32
                # Draw crosshair
                painter.drawLine(self.roi_marker.x() - 40, self.roi_marker.y(),
                               self.roi_marker.x() + 40, self.roi_marker.y())
                painter.drawLine(self.roi_marker.x(), self.roi_marker.y() - 40,
                               self.roi_marker.x(), self.roi_marker.y() + 40)
                # Draw ROI boundary square
                painter.drawRect(self.roi_marker.x() - 32, self.roi_marker.y() - 32, 64, 64)
            
            painter.end()
            
            # Set the pixmap to the label
            self.image_label.setPixmap(pixmap)
            self.image_label.setFixedSize(pixmap.size())
            
            # Adjust window size on first image
            if not hasattr(self, 'window_sized'):
                screen = QApplication.primaryScreen().availableGeometry()
                total_width = min(w + self.right_panel.width() + 50, screen.width() - 100)
                total_height = min(h + self.status_label.height() + 80, screen.height() - 100)
                
                self.resize(total_width, total_height)
                
                frame_geo = self.frameGeometry()
                screen_center = screen.center()
                frame_geo.moveCenter(screen_center)
                self.move(frame_geo.topLeft())
                
                self.window_sized = True
                
        except Exception as e:
            logger.error(f"Display update error: {e}")
            self.show_error(f"Display update error: {e}")
    
    def update_status(self, message, msg_type="info"):
        """Update the status label with appropriate styling"""
        self.status_label.setText(message)
        
        if msg_type == "error":
            self.status_label.setStyleSheet("background-color: #FFCCCC; color: #CC0000; padding: 5px;")
        elif msg_type == "warning":
            self.status_label.setStyleSheet("background-color: #FFFFCC; color: #CC6600; padding: 5px;")
        else:  # info
            self.status_label.setStyleSheet("background-color: #CCFFCC; color: #006600; padding: 5px;")
    
    def update_arm_status(self, message, msg_type="info"):
        """Update arm status label"""
        self.arm_status_label.setText(f"Arm: {message}")
        
        if msg_type == "error":
            self.arm_status_label.setStyleSheet("background-color: #FFCCCC; color: #CC0000; padding: 5px;")
        elif msg_type == "warning":
            self.arm_status_label.setStyleSheet("background-color: #FFFFCC; color: #CC6600; padding: 5px;")
        else:  # info
            self.arm_status_label.setStyleSheet("background-color: #FFE6CC; color: #FF6600; padding: 5px;")
    
    def update_arm_position(self, x_steps, y_steps, z_steps):
        """Update arm position display"""
        self.update_arm_status(f"Position - X:{x_steps}, Y:{y_steps}, Z:{z_steps}", "info")
    
    def update_arm_connection_status(self, success, x_state, y_state, z_state):
        """Update arm connection status display"""
        if success:
            self.update_arm_status(f"Connection - X:{x_state}, Y:{y_state}, Z:{z_state}", "info")
        else:
            self.update_arm_status("Connection check failed", "error")
    
    def update_pump_status(self, message, msg_type="info"):
        """Update pump status label"""
        self.pump_status_label.setText(f"Pump: {message}")
        
        if msg_type == "error":
            self.pump_status_label.setStyleSheet("background-color: #FFCCCC; color: #CC0000; padding: 5px;")
        elif msg_type == "warning":
            self.pump_status_label.setStyleSheet("background-color: #FFFFCC; color: #CC6600; padding: 5px;")
        else:  # info
            self.pump_status_label.setStyleSheet("background-color: #E6F3FF; color: #0066CC; padding: 5px;")
    
    def update_pump_position(self, x, y, z):
        """Update pump position display"""
        self.update_pump_status(f"Position - X:{x:.2f}, Y:{y:.2f}, Z:{z:.2f} μm", "info")
    
    def update_recording_status(self, message, msg_type="info"):
        """Update recording status label"""
        self.recording_status_label.setText(message)
        
        if msg_type == "error":
            self.recording_status_label.setStyleSheet("background-color: #FFCCCC; color: #CC0000; padding: 5px;")
        elif msg_type == "warning":
            self.recording_status_label.setStyleSheet("background-color: #FFFFCC; color: #CC6600; padding: 5px;")
        else:  # info
            self.recording_status_label.setStyleSheet("background-color: #CCFFCC; color: #006600; padding: 5px;")
    
    def update_dataset_progress(self, current, total):
        """Update dataset collection progress"""
        self.dataset_progress_label.setText(f"Collecting depth images: {current}/{total}")
        self.dataset_progress_label.show()
    
    def show_error(self, error_message):
        """Show error message"""
        self.update_status(f"ERROR: {error_message}", "error")
        logger.error(error_message)
    
    # Video recording methods
    def start_recording(self):
        """Start video recording"""
        if not self.is_recording:
            width, height = self.current_image_size
            success = self.video_recording_thread.start_recording(width, height, 30)
            
            if success:
                self.is_recording = True
                self.start_recording_btn.setEnabled(False)
                self.stop_recording_btn.setEnabled(True)
                self.update_recording_status("Recording...", "warning")
    
    def stop_recording(self):
        """Stop video recording"""
        if self.is_recording:
            success = self.video_recording_thread.stop_recording()
            
            if success:
                self.is_recording = False
                self.start_recording_btn.setEnabled(True)
                self.stop_recording_btn.setEnabled(False)
                self.update_recording_status("Recording stopped", "info")
    
    def on_recording_started(self):
        """Handle recording started signal"""
        self.update_recording_status("Recording in progress...", "warning")
    
    def on_recording_stopped(self, filename):
        """Handle recording stopped signal"""
        if filename:
            self.update_recording_status(f"Video saved: {os.path.basename(filename)}", "info")
        else:
            self.update_recording_status("Recording stopped", "info")
    
    # Stage control methods
    def start_calibration(self):
        """Start the stage calibration process"""
        self.calibration_mode = True
        self.reference_point_mode = False
        self.depth_collection_mode = False
        self.calibration_point_index = -1
        self.current_calibration_point = None
        self.stage_thread.request_calibration()
    
    def start_set_reference_point(self):
        """Start the process of setting a reference point"""
        if not self.stage_thread.is_calibrated:
            self.update_status("Stage not calibrated. Please calibrate first.", "warning")
            return
            
        self.reference_point_mode = True
        self.calibration_mode = False
        self.depth_collection_mode = False
        self.update_status("Click on the image to set the reference point", "warning")
    
    def start_collect_z_dataset(self):
        """Start Z depth dataset collection mode"""
        self.depth_collection_mode = True
        self.calibration_mode = False
        self.reference_point_mode = False
        self.update_status("Double-click on the ball position in the image to start Z-depth dataset collection", "warning")
        # Disable the button during collection
        self.collect_z_dataset_btn.setEnabled(False)
    
    def show_calibration_point(self, index, point):
        """Show a calibration point"""
        self.calibration_mode = True
        self.calibration_point_index = index
        self.current_calibration_point = point
    
    def show_roi_marker(self, center_point):
        """Show ROI marker for depth dataset collection"""
        self.roi_marker = center_point
        # Set timer to clear ROI marker after 10 seconds (enough time for collection)
        self.marker_timer.start(10000)
    
    def on_calibration_complete(self, success):
        """Handle calibration completion"""
        self.calibration_mode = False
        if success:
            self.set_ref_point_btn.setEnabled(True)
    
    def on_dataset_collection_complete(self, success, message):
        """Handle depth dataset collection completion"""
        self.depth_collection_mode = False
        self.dataset_progress_label.hide()
        
        # Re-enable the collect button
        self.collect_z_dataset_btn.setEnabled(True)
        
        # Clear ROI marker
        self.roi_marker = None
        
        if success:
            self.update_status(f"Z-depth dataset collection completed: {message}", "info")
        else:
            self.update_status(f"Z-depth dataset collection failed: {message}", "error")
    
    # Arm control methods
    def check_arm_connection(self):
        """Check arm connection status"""
        self.arm_thread.request_check_connection()
        self.update_arm_status("Checking connection...", "info")
    
    def get_arm_position(self):
        """Get current arm position"""
        self.arm_thread.request_get_position()
        self.update_arm_status("Getting position...", "info")
    
    def move_arm(self, motor_type, speed, step):
        """Move arm motor"""
        self.arm_thread.request_move_motor(motor_type, speed, step)
        motor_names = {1: "X", 2: "Y", 3: "Z"}
        motor_name = motor_names.get(motor_type, "Unknown")
        self.update_arm_status(f"Moving {motor_name} motor by {step} steps at speed {speed}", "info")
    
    # Pump control methods
    def pump_step_cw(self):
        """Pump step clockwise"""
        step = self.pump_step_spin.value()
        self.pump_thread.request_step_cw(step)
        self.update_pump_status(f"Stepping CW by {step} steps", "info")
    
    def pump_step_ccw(self):
        """Pump step counter-clockwise"""
        step = self.pump_step_spin.value()
        self.pump_thread.request_step_ccw(step)
        self.update_pump_status(f"Stepping CCW by {step} steps", "info")
    
    def pump_set_speed(self):
        """Set pump speed"""
        speed = self.pump_speed_spin.value()
        self.pump_thread.request_set_speed(speed)
        self.update_pump_status(f"Speed set to {speed} RPM", "info")
    
    def pump_run(self):
        """Run pump continuously"""
        speed = self.pump_speed_spin.value()
        self.pump_thread.request_run(speed)
        self.update_pump_status(f"Running continuously at {speed} RPM", "info")
    
    # Mouse and keyboard event handlers
    def image_click(self, event):
        """Handle mouse click events on the image"""
        pos = event.position()
        click_x = int(pos.x())
        click_y = int(pos.y())
        click_point = QPoint(click_x, click_y)
        
        if self.calibration_mode:
            # Add calibration point
            self.stage_thread.request_add_calibration_point(click_point, self.calibration_point_index)
            self.current_calibration_point = None
            
        elif self.reference_point_mode:
            # Set reference point
            success = self.stage_thread.set_reference_point(click_x, click_y)
            if success:
                self.reference_marker = click_point
                self.update_status(f"Reference point set to ({click_x}, {click_y})", "info")
            
            self.reference_point_mode = False
    
    def image_double_click(self, event):
        """Handle mouse double-click events on the image"""
        pos = event.position()
        click_x = int(pos.x())
        click_y = int(pos.y())
        click_point = QPoint(click_x, click_y)
        
        if self.depth_collection_mode:
            # Start depth dataset collection
            self.depth_dataset_thread.request_collect_dataset(click_x, click_y)
            self.update_status(f"Starting Z-depth dataset collection at ({click_x}, {click_y})", "info")
            return
        
        # If in calibration or reference point mode, ignore double-clicks
        if self.calibration_mode or self.reference_point_mode:
            return
            
        # Check if stage is calibrated
        if not self.stage_thread.is_calibrated:
            self.update_status("Stage not calibrated. Please click 'Calibrate Stage' button first.", "warning")
            return
            
        # Check if reference point is set
        if self.stage_thread.reference_point is None:
            self.update_status("Reference point not set. Please click 'Set Reference Point' button first.", "warning")
            return
        
        # Set target marker
        self.target_marker = click_point
        
        # Request stage movement
        self.stage_thread.request_move_to(click_x, click_y)
        self.update_status(f"Moving stage to bring point ({click_x}, {click_y}) to reference position", "info")
        
        # Set timer to clear marker after 300ms
        self.marker_timer.start(300)
    
    def image_wheel_event(self, event):
        """Handle mouse wheel events for Z-axis control"""
        delta = event.angleDelta().y()
        
        if delta > 0:  # Scroll up
            z_delta = self.z_step
            self.update_status(f"Z-axis up {self.z_step} μm", "info")
        else:  # Scroll down
            z_delta = -self.z_step
            self.update_status(f"Z-axis down {self.z_step} μm", "info")
        
        # Request Z-axis movement
        self.stage_thread.request_move_z_relative(z_delta)
        
        event.accept()
    
    def toggle_detection(self):
        """Toggle YOLO detection and tracking"""
        if not self.yolo_detecting:
            # Start detection
            self.yolo_detecting = True
            self.yolo_thread.start_detection()
            self.detect_track_btn.setText("Stop Detect & Track")
            self.detect_track_btn.setStyleSheet("background-color: #FF9999; font-weight: bold; font-size: 14px;")
            self.detection_status_label.setText("Detection: ON")
            self.detection_status_label.setStyleSheet("background-color: #E6FFE6; color: #006600; padding: 5px;")
        else:
            # Stop detection
            self.yolo_detecting = False
            self.yolo_thread.stop_detection()
            self.detect_track_btn.setText("Start Detect & Track")
            self.detect_track_btn.setStyleSheet("background-color: #99CCFF; font-weight: bold; font-size: 14px;")
            self.detection_status_label.setText("Detection: OFF")
            self.detection_status_label.setStyleSheet("background-color: #FFE6E6; color: #CC0000; padding: 5px;")
            self.detection_overlay = None
            
            # Clear FPS display
            self.detection_fps_label.setText("Detection FPS: --")
            self.tracking_fps_label.setText("Tracking FPS: --")
    
    def update_detection_overlay(self, frame, boxes, scores, class_ids, tracks):
        """Update detection overlay data"""
        self.detection_overlay = (boxes, scores, class_ids, tracks)
    
    def update_detection_status(self, message, msg_type):
        """Update detection status message"""
        if msg_type == "error":
            logger.error(f"YOLO: {message}")
            # Show error in main status if critical
            if "initialization failed" in message.lower():
                self.update_status(f"YOLO Error: {message}", "error")
        elif msg_type == "warning":
            logger.warning(f"YOLO: {message}")
        else:
            logger.info(f"YOLO: {message}")
    
    def update_fps_display(self, detection_fps, tracking_fps):
        """Update FPS display"""
        self.detection_fps_label.setText(f"Detection FPS: {detection_fps:.1f}")
        self.tracking_fps_label.setText(f"Tracking FPS: {tracking_fps:.1f}")
    
    def update_frame_skip(self, value):
        """Update frame skip setting"""
        if self.yolo_thread:
            self.yolo_thread.set_skip_frames(value)

    def clear_markers(self):
        """Clear visual markers after timer expires"""
        if self.marker_timer.isActive():
            self.marker_timer.stop()
        
        if not self.calibration_mode:
            self.current_calibration_point = None
            
        self.target_marker = None
        
        # Don't clear ROI marker automatically during dataset collection
        if not self.depth_collection_mode:
            self.roi_marker = None
    
    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Stop all threads
        self.camera_thread.stop()
        self.stage_thread.stop()
        self.arm_thread.stop()
        self.pump_thread.stop()
        self.video_recording_thread.stop()
        self.depth_dataset_thread.stop()
        # Stop YOLO thread
        if self.yolo_detecting:
            self.toggle_detection()
        self.yolo_thread.stop()
        
        # Wait for threads to finish
        self.camera_thread.wait()
        self.stage_thread.wait()
        self.arm_thread.wait()
        self.pump_thread.wait()
        self.video_recording_thread.wait()
        self.depth_dataset_thread.wait()
        self.yolo_thread.wait()
        
        # Accept the close event
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()