'''
python val_video_tensorrt_with_tracking.py --video ~/UNeXt-pytorch/inputs/Sperm_Selection_Video_Test/20241105_121237814.mp4 --output video_tensorrt_results/20241105_121237814_tracking.mp4 --conf 0.25 
python val_video_tensorrt_with_tracking.py --video ~/UNeXt-pytorch/inputs/Sperm_Selection_Video_Test/20241105_115540765.mp4 --output video_tensorrt_results/20241105_115540765_tracking.mp4 --conf 0.25 

python val_video_tensorrt_with_tracking.py --video input_video.mp4 --output results/output_video.mp4 --conf 0.25 --preview
'''

import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from pathlib import Path
import json
import argparse
from collections import defaultdict
import colorsys

# 导入JPDAF跟踪器
try:
    from jpdaf_tracker import JPDAFilter
except ImportError:
    print("警告: 无法导入JPDAF跟踪器，跟踪功能将被禁用")
    JPDAFilter = None

# 检查环境
print(f"使用GPU: {cuda.Device(0).name()}")
print(f"CUDA版本: {cuda.get_version()}")
print(f"TensorRT版本: {trt.__version__}")

class TensorRTInference:
    def __init__(self, engine_path):
        """初始化TensorRT推理引擎"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        print("正在加载TensorRT引擎...")
        
        # 检查引擎文件是否存在
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT引擎文件不存在: {engine_path}")
        
        # 加载引擎
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("无法加载TensorRT引擎")
        
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("无法创建TensorRT执行上下文")
        
        # 获取输入输出信息 - 兼容新旧API
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        # 检查TensorRT版本并使用相应的API
        try:
            # 新版本API (TensorRT 8+)
            num_io_tensors = self.engine.num_io_tensors
            tensor_names = [self.engine.get_tensor_name(i) for i in range(num_io_tensors)]
            
            for name in tensor_names:
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = self.engine.get_tensor_shape(name)
                
                # 处理动态形状
                if -1 in shape:
                    # 设置默认输入形状
                    if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        shape = (1, 3, 640, 640)  # 默认输入形状
                        self.context.set_input_shape(name, shape)
                
                size = trt.volume(shape)
                
                # 分配设备内存
                device_mem = cuda.mem_alloc(size * dtype().itemsize)
                self.bindings.append(int(device_mem))
                
                # 分配主机内存
                host_mem = cuda.pagelocked_empty(size, dtype)
                
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.inputs.append({'name': name, 'shape': shape, 'dtype': dtype, 
                                      'host': host_mem, 'device': device_mem})
                else:
                    self.outputs.append({'name': name, 'shape': shape, 'dtype': dtype, 
                                       'host': host_mem, 'device': device_mem})
                    
        except AttributeError:
            # 旧版本API (TensorRT 7.x)
            print("使用旧版本TensorRT API")
            num_bindings = self.engine.num_bindings
            
            for i in range(num_bindings):
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = self.engine.get_binding_shape(i)
                size = trt.volume(shape)
                
                # 分配设备内存
                device_mem = cuda.mem_alloc(size * dtype().itemsize)
                self.bindings.append(int(device_mem))
                
                # 分配主机内存
                host_mem = cuda.pagelocked_empty(size, dtype)
                
                if self.engine.binding_is_input(i):
                    self.inputs.append({'name': name, 'shape': shape, 'dtype': dtype, 
                                      'host': host_mem, 'device': device_mem})
                else:
                    self.outputs.append({'name': name, 'shape': shape, 'dtype': dtype, 
                                       'host': host_mem, 'device': device_mem})
        
        print("TensorRT引擎加载成功")
        print(f"输入形状: {self.inputs[0]['shape']}")
        print(f"输入数据类型: {self.inputs[0]['dtype']}")
        print(f"输出数量: {len(self.outputs)}")
    
    def preprocess(self, img):
        """预处理图片(接收opencv图像)"""
        if img is None:
            raise ValueError("无法处理空图像")
        
        self.original_shape = img.shape[:2]  # 保存原始尺寸
        
        # 转换颜色空间 BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整大小到640x640
        img_resized = cv2.resize(img, (640, 640))
        
        # 归一化到[0,1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # 转换为CHW格式
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        
        # 添加batch维度
        img_batch = np.expand_dims(img_chw, axis=0)
        
        # 确保连续内存并转换为正确的数据类型
        img_batch = np.ascontiguousarray(img_batch)
        
        # 转换为模型所需的数据类型
        if self.inputs[0]['dtype'] != img_batch.dtype:
            img_batch = img_batch.astype(self.inputs[0]['dtype'])
            
        return img_batch
    
    def infer(self, input_data):
        """执行推理"""
        # 确保输入数据与输入形状匹配
        expected_size = np.prod(self.inputs[0]['shape'])
        input_size = input_data.size
        
        if input_size != expected_size:
            print(f"输入数据大小不匹配: 预期 {expected_size}，实际 {input_size}")
            # 确保输入数据大小正确
            input_data = input_data.reshape(self.inputs[0]['shape'])
        
        # 确保数据类型正确
        if input_data.dtype != self.inputs[0]['dtype']:
            print(f"输入数据类型不匹配: 预期 {self.inputs[0]['dtype']}，实际 {input_data.dtype}")
            input_data = input_data.astype(self.inputs[0]['dtype'])
        
        # 复制输入数据到设备
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 设置输入张量地址
        try:
            # 新版本API
            for i, inp in enumerate(self.inputs):
                self.context.set_tensor_address(inp['name'], inp['device'])
            for i, out in enumerate(self.outputs):
                self.context.set_tensor_address(out['name'], out['device'])
            
            # 执行推理
            self.context.execute_async_v3(stream_handle=self.stream.handle)
        except AttributeError:
            # 旧版本API
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 复制输出数据到主机
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
        
        # 同步
        self.stream.synchronize()
        
        # 重塑输出
        output = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
        return output
    
    def postprocess(self, output, conf_threshold=0.25, iou_threshold=0.45):
        """后处理 - 适配YOLOv8输出格式"""
        # YOLOv8输出格式: [1, 84, 8400] 或 [1, num_classes+4, num_boxes]
        # 其中前4个是bbox坐标，后面是各类别的置信度
        
        # 移除batch维度
        if output.ndim == 3:
            output = output[0]  # [84, 8400] 或 [num_classes+4, num_boxes]
        
        # 转置以获得 [num_boxes, num_classes+4] 格式
        predictions = output.T
        
        # 提取边界框和分数
        boxes = predictions[:, :4]  # x, y, w, h
        scores = predictions[:, 4:]  # 所有类别的分数
        
        # 获取每个框的最高分数和对应类别
        class_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        # 过滤低置信度的检测
        mask = class_scores > conf_threshold
        if not mask.any():
            return [], [], []
        
        boxes = boxes[mask]
        class_scores = class_scores[mask]
        class_ids = class_ids[mask]
        
        # 转换边界框格式：YOLO格式(cx, cy, w, h) -> (x1, y1, x2, y2)
        boxes_xyxy = self.xywh2xyxy(boxes)
        
        # 缩放到原始图像尺寸
        scale_x = self.original_shape[1] / 640
        scale_y = self.original_shape[0] / 640
        
        boxes_xyxy[:, [0, 2]] *= scale_x
        boxes_xyxy[:, [1, 3]] *= scale_y
        
        # NMS
        indices = self.nms(boxes_xyxy, class_scores, iou_threshold)
        
        return boxes_xyxy[indices], class_scores[indices], class_ids[indices]
    
    def xywh2xyxy(self, boxes):
        """转换边界框格式"""
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return boxes_xyxy
    
    def nms(self, boxes, scores, iou_threshold):
        """非极大值抑制"""
        if len(boxes) == 0:
            return []
        
        # 计算面积
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # 按分数排序
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # 计算IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            # 保留IoU小于阈值的框
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep

def get_track_colors(num_tracks):
    """生成跟踪目标的颜色"""
    colors = []
    for i in range(num_tracks):
        # 使用HSV色彩空间生成均匀分布的颜色
        h = i / num_tracks
        s = 0.8
        v = 0.8
        rgb = colorsys.hsv_to_rgb(h, s, v)
        rgb = tuple(int(x * 255) for x in rgb)
        colors.append(rgb)
    return colors

def draw_boxes_and_tracks(img, boxes, scores, class_ids, class_names, tracker=None):
    """在图片上绘制检测框和轨迹，返回绘制后的图片"""
    # 复制图片以避免修改原图
    img_with_boxes = img.copy()
    
    # 生成颜色
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                   (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)]
    
    # 先绘制跟踪轨迹
    if tracker is not None:
        try:
            active_tracks = tracker.get_active_tracks()
            track_colors = get_track_colors(100)  # 预生成100种颜色
            
            for track in active_tracks:
                # 只绘制有足够长轨迹的目标
                if len(track.trajectory) >= 2:
                    color = track_colors[track.id % len(track_colors)]
                    
                    # 绘制轨迹线
                    for i in range(1, len(track.trajectory)):
                        pt1 = (int(track.trajectory[i-1][0]), int(track.trajectory[i-1][1]))
                        pt2 = (int(track.trajectory[i][0]), int(track.trajectory[i][1]))
                        cv2.line(img_with_boxes, pt1, pt2, color, 2)
                    
                    # 绘制轨迹ID
                    last_pt = track.trajectory[-1]
                    cv2.putText(img_with_boxes, f"ID:{track.id}", 
                               (int(last_pt[0]), int(last_pt[1]) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(f"绘制轨迹时出错: {e}")
    
    # 然后绘制检测框
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        
        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_with_boxes.shape[1], x2)
        y2 = min(img_with_boxes.shape[0], y2)
        
        # 安全地获取类别ID对应的颜色
        color_idx = int(class_id) % len(base_colors)
        color = base_colors[color_idx]
        
        # 绘制边界框
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本 - 安全地访问类别名称
        try:
            if int(class_id) < len(class_names):
                label = f"{class_names[int(class_id)]}: {score:.2f}"
            else:
                label = f"class_{int(class_id)}: {score:.2f}"
        except (IndexError, ValueError):
            label = f"unknown: {score:.2f}"
        
        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # 绘制文本背景
        cv2.rectangle(img_with_boxes, (x1, y1 - text_height - baseline), 
                     (x1 + text_width, y1), color, -1)
        
        # 绘制文本
        cv2.putText(img_with_boxes, label, (x1, y1 - baseline), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_with_boxes

def process_video(engine, video_path, output_path, class_names, conf_threshold=0.25, iou_threshold=0.45, show_preview=False, enable_tracking=True):
    """处理视频文件"""
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height} @ {fps}fps, 共 {total_frames} 帧")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 初始化跟踪器
    tracker = None
    if enable_tracking and JPDAFilter is not None:
        try:
            tracker = JPDAFilter(process_noise=20.0, measure_noise=2.0, detect_prob=0.7, gate_prob=0.95)
        except Exception as e:
            print(f"无法初始化跟踪器: {e}")
            tracker = None
    
    # 性能统计
    inference_times = []
    processing_times = []
    frame_count = 0
    total_detections = 0
    
    # 处理视频帧
    print("开始处理视频...")
    start_time = time.time()
    
    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 显示进度
        if frame_count % 10 == 0 or frame_count == 1:
            progress = (frame_count / total_frames) * 100
            print(f"处理进度: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        # 计时开始
        frame_start_time = time.time()
        
        # 预处理
        input_data = engine.preprocess(frame)
        
        # 推理
        infer_start_time = time.time()
        output = engine.infer(input_data)
        infer_time = (time.time() - infer_start_time) * 1000  # 转为毫秒
        
        # 后处理
        boxes, scores, class_ids = engine.postprocess(output, conf_threshold, iou_threshold)
        
        # 跟踪检测结果
        if enable_tracking and tracker is not None:
            try:
                # 将边界框转换为跟踪点(使用中心点)
                detection_points = []
                for box in boxes:
                    x_center = (box[0] + box[2]) / 2
                    y_center = (box[1] + box[3]) / 2
                    detection_points.append((x_center, y_center))
                
                # 更新跟踪器
                tracker.predict()
                tracker.correct(detection_points)
            except Exception as e:
                print(f"跟踪更新失败: {e}")
        
        # 计时结束
        frame_time = (time.time() - frame_start_time) * 1000  # 转为毫秒
        
        # 统计
        inference_times.append(infer_time)
        processing_times.append(frame_time)
        total_detections += len(boxes)
        
        # 绘制检测结果和轨迹
        result_frame = draw_boxes_and_tracks(frame, boxes, scores, class_ids, class_names, tracker)
        
        # 添加帧处理信息
        info_text = f"frame: {frame_count}/{total_frames} | inference: {infer_time:.1f}ms | total time: {frame_time:.1f}ms | detection: {len(boxes)}"
        cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 写入输出视频
        out.write(result_frame)
        
        # 显示预览（可选）
        if show_preview:
            # 调整预览窗口大小（如果原始帧太大）
            preview_frame = result_frame
            if width > 1280 or height > 720:
                scale = min(1280 / width, 720 / height)
                preview_frame = cv2.resize(result_frame, (int(width * scale), int(height * scale)))
            
            cv2.imshow('YOLOv8 TensorRT Detection & Tracking', preview_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # 计算总处理时间
    total_time = time.time() - start_time
    
    # 释放资源
    cap.release()
    out.release()
    if show_preview:
        cv2.destroyAllWindows()
    
    # 计算统计信息
    if inference_times:
        avg_inference_time = np.mean(inference_times)
        avg_processing_time = np.mean(processing_times)
        fps_achieved = 1000 / avg_processing_time
    else:
        avg_inference_time = 0
        avg_processing_time = 0
        fps_achieved = 0
    
    # 生成报告
    report = f"""
==================================================
TensorRT 视频处理性能报告:
==================================================
视频文件: {video_path}
输出文件: {output_path}
总帧数: {frame_count}
总检测数: {total_detections}
平均每帧检测数: {total_detections/frame_count:.2f} 

时间统计:
- 总处理时间: {total_time:.2f} 秒
- 平均推理时间: {avg_inference_time:.2f} ms/帧
- 平均处理时间: {avg_processing_time:.2f} ms/帧 (包含预处理和后处理)
- 实际处理帧率: {fps_achieved:.2f} FPS

原始视频信息:
- 分辨率: {width}x{height}
- 帧率: {fps} FPS
"""
    
    print(report)
    
    # 保存报告
    report_path = Path(output_path).with_suffix('.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"视频处理完成！结果已保存到: {output_path}")
    print(f"性能报告已保存到: {report_path}")
    
    return {
        'frames_processed': frame_count,
        'total_detections': total_detections,
        'avg_inference_time': avg_inference_time,
        'avg_processing_time': avg_processing_time,
        'achieved_fps': fps_achieved
    }

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLOv8 TensorRT 视频处理与目标跟踪')
    parser.add_argument('--engine', type=str, default="weights/yolov8-weights/best.engine", help='TensorRT引擎文件路径')
    parser.add_argument('--video', type=str, required=True, help='输入视频文件路径')
    parser.add_argument('--output', type=str, help='输出视频文件路径 (默认在同目录下添加_processed后缀)')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--preview', action='store_true', help='实时显示处理预览窗口')
    parser.add_argument('--no-tracking', action='store_true', help='禁用目标跟踪')
    args = parser.parse_args()
    
    # 设置默认输出路径（如果未指定）
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.with_stem(f"{video_path.stem}_processed").with_suffix('.mp4'))
    
    # 创建输出目录
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 类别名称（根据您的数据集调整）
    class_names = [
        'ball_contact',
        'ball_free', 
        'ball_inside',
        'ball_under',
        'pipette_tip'
    ]
    
    # 检查引擎文件是否存在
    if not os.path.exists(args.engine):
        print(f"错误：TensorRT引擎文件不存在: {args.engine}")
        print("请先使用pt2engine.py转换模型")
        return
    
    try:
        # 初始化推理引擎
        engine = TensorRTInference(args.engine)
        
        # 预热
        print("正在进行预热运行...")
        try:
            # 使用模型输入的正确数据类型创建预热输入
            input_dtype = engine.inputs[0]['dtype']
            dummy_input = np.random.randn(1, 3, 640, 640).astype(input_dtype)
            for _ in range(5):
                engine.infer(dummy_input)
            print("预热完成！\n")
        except Exception as e:
            print(f"预热过程中出错: {e}")
            print("继续执行，但性能可能受到影响")
        
        # 处理视频
        process_video(engine, args.video, args.output, class_names, 
                     conf_threshold=args.conf, iou_threshold=args.iou, 
                     show_preview=args.preview, enable_tracking=not args.no_tracking)
                     
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()