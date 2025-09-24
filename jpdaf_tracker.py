import numpy as np
import cv2
from scipy.stats import chi2

class JPDATrack:
    """JPDA单个跟踪目标类"""
    
    # 20.0, 2.0, 0.7, 0.95
    def __init__(self, pt, process_noise=2.0, measure_noise=20.0, pd=0.7, pg=0.8, track_id=None):
        """初始化跟踪目标
        
        Args:
            pt: 初始点坐标 (x, y)
            process_noise: 过程噪声协方差
            measure_noise: 测量噪声协方差
            pd: 检测概率
            pg: 门限概率
            track_id: 轨迹ID
        """
        self.id = track_id
        self.isActive = True
        self.Pd = pd
        self.Pg = pg
        self.GateThreshold = -2.0 * np.log(1 - pg)  # 马氏距离阈值
        
        # 初始化卡尔曼滤波器状态
        dt = 1.0 / 30.0  # 假设30fps
        
        # 状态转移矩阵 (x, y, vx, vy)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        # 测量矩阵 (只测量位置x, y)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        
        # 初始状态
        self.x_pred = np.zeros((4, 1), dtype=np.float32)  # 预测状态
        self.x_post = np.array([[pt[0]], [pt[1]], [0], [0]], dtype=np.float32)  # 后验状态
        
        # 状态协方差
        self.P_pred = np.zeros((4, 4), dtype=np.float32)  # 预测状态协方差
        self.P_post = np.eye(4, dtype=np.float32)  # 后验状态协方差
        
        # 过程噪声协方差
        self.Q0 = np.eye(4, dtype=np.float32) * process_noise
        self.Q_pred = self.Q0.copy()
        self.Q_post = self.Q0.copy()
        
        # 测量噪声协方差
        self.R = np.eye(2, dtype=np.float32) * measure_noise
        
        # 轨迹点列表
        self.trajectory = [pt]
        
        # 跟踪帧计数
        self.age = 0
        self.consecutive_misses = 0
        
    def predict(self):
        """预测状态"""
        # 预测状态
        self.x_pred = self.F @ self.x_post
        
        # 预测测量
        self.z_pred = self.H @ self.x_post
        
        # 自适应过程噪声
        v = self.x_pred - self.x_post
        self.Q_pred = 0.8 * self.Q_post + 0.2 * self.Q0
        self.Q_post = self.Q_pred
        
        # 预测状态协方差
        self.P_pred = self.F @ self.P_post @ self.F.T + self.Q_pred
        
        # 计算新息协方差
        self.S = self.H @ self.P_pred @ self.H.T + self.R
        
        # 计算卡尔曼增益
        self.invS = np.linalg.inv(self.S)
        self.K = self.P_pred @ self.H.T @ self.invS
        
        # 计算门限区域体积
        self.detS = np.linalg.det(self.S)
        self.gatingVolume = np.pi * self.GateThreshold * np.sqrt(self.detS)
        
        # 计算门限椭圆
        self.gatingEllipse = self.get_error_ellipse(self.GateThreshold, 
                                                   (self.z_pred[0, 0], self.z_pred[1, 0]), 
                                                   self.S)
        
        # 记录原始尺寸，用于后处理
        self.original_shape = None
        
        # 更新轨迹年龄
        self.age += 1
        
    def get_error_ellipse(self, chisquare_val, mean, covmat):
        """计算误差椭圆"""
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(covmat)
        
        # 计算椭圆主轴方向
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        
        # 转换为度
        angle = 180.0 * angle / np.pi
        
        # 计算主轴和次轴大小
        major_axis = 2.0 * chisquare_val * np.sqrt(eigenvalues[0])
        minor_axis = 2.0 * chisquare_val * np.sqrt(eigenvalues[1])
        
        # 返回旋转矩形
        return ((mean[0], mean[1]), (major_axis, minor_axis), -angle)
    
    def calc_mahal_distance(self, measure):
        """计算马氏距离"""
        dx = measure[0] - self.z_pred[0, 0]
        dy = measure[1] - self.z_pred[1, 0]
        
        a00 = self.invS[0, 0]
        a01 = self.invS[0, 1]
        a10 = self.invS[1, 0]
        a11 = self.invS[1, 1]
        
        res_val = dx * (dx * a00 + dy * a10) + dy * (dx * a01 + dy * a11)
        return np.sqrt(res_val)
    
    def calc_euclidean_distance(self, measure):
        """计算欧氏距离"""
        dx = measure[0] - self.z_pred[0, 0]
        dy = measure[1] - self.z_pred[1, 0]
        return np.sqrt(dx * dx + dy * dy)
    
    def correct(self, valid_measures, betta):
        """基于有效测量修正状态
        
        Args:
            valid_measures: 有效测量列表 [(x1,y1), (x2,y2), ...]
            betta: 关联概率列表 [b1, b2, ...]
        """
        # 如果没有有效测量，使用预测状态
        if len(valid_measures) == 0:
            self.x_post = self.x_pred
            self.P_post = self.P_pred
            self.consecutive_misses += 1
            return
        
        self.consecutive_misses = 0
        
        # 计算betta0
        betta0 = 1.0 - sum(betta)
        
        # 1. 状态估计
        error_combined = np.zeros((2, 1), dtype=np.float32)
        for i, measure in enumerate(valid_measures):
            error_i = np.array([[measure[0] - self.z_pred[0, 0]],
                               [measure[1] - self.z_pred[1, 0]]], dtype=np.float32)
            error_combined += betta[i] * error_i
        
        self.x_post = self.x_pred + self.K @ error_combined
        
        # 2. 更新状态协方差
        Pc = (np.eye(4) - self.K @ self.H) @ self.P_pred
        
        temp1 = np.zeros((2, 2), dtype=np.float32)
        for i, measure in enumerate(valid_measures):
            error_i = np.array([[measure[0] - self.z_pred[0, 0]],
                               [measure[1] - self.z_pred[1, 0]]], dtype=np.float32)
            temp1 += betta[i] * error_i @ error_i.T
        
        PP = self.K @ (temp1 - error_combined @ error_combined.T) @ self.K.T
        
        self.P_post = betta0 * self.P_pred + (1.0 - betta0) * Pc + PP
        
        # 更新轨迹
        self.trajectory.append((self.x_post[0, 0], self.x_post[1, 0]))
        

class JPDAFilter:
    """JPDA滤波器类"""
    
    def __init__(self, process_noise=20.0, measure_noise=2.0, detect_prob=0.7, gate_prob=0.95):
        """初始化JPDA滤波器
        
        Args:
            process_noise: 过程噪声协方差
            measure_noise: 测量噪声协方差
            detect_prob: 检测概率
            gate_prob: 门限概率
        """
        self.processNoise = process_noise
        self.measureNoise = measure_noise
        self.Pd = detect_prob
        self.Pg = gate_prob
        self.GateThreshold = -2.0 * np.log(1 - gate_prob)
        self.trackID = 0
        self.tracks = []
        
    def reset(self):
        """重置跟踪器"""
        self.tracks = []
        self.trackID = 0
        
    def predict(self):
        """预测所有轨迹状态"""
        for track in self.tracks:
            track.predict()
    
    def correct(self, all_measurements):
        """修正轨迹状态
        
        Args:
            all_measurements: 所有测量点列表 [(x1,y1), (x2,y2), ...]
        """
        self.validate_measurements(all_measurements)
        self.cluster_measurements()
        self.correct_each_cluster(all_measurements)
        self.delete_tracks()
        self.init_unassociate_tracks()
    
    def validate_measurements(self, all_measurements):
        """验证测量点
        
        Args:
            all_measurements: 所有测量点列表
        """
        n_tracks = len(self.tracks)
        n_measure = len(all_measurements)
        
        self.validationMatrix = np.zeros((n_tracks, n_measure), dtype=np.uint8)
        self.distanceMatrix = np.zeros((n_tracks, n_measure), dtype=np.float32)
        self.unValidatedMeasures = []
        
        temp_dist_e_threshold = 25.0
        
        for i, measure in enumerate(all_measurements):
            is_associated = False
            for j, track in enumerate(self.tracks):
                temp_dist = track.calc_mahal_distance(measure)
                self.distanceMatrix[j, i] = temp_dist
                temp_dist_e = track.calc_euclidean_distance(measure)
                
                if temp_dist < self.GateThreshold:  # and temp_dist_e < temp_dist_e_threshold:
                    is_associated = True
                    self.validationMatrix[j, i] = 1
                    
            if not is_associated:
                self.unValidatedMeasures.append(measure)
    
    def cluster_measurements(self):
        """聚类测量点"""
        # 1. 获取所有有效测量
        all_ones = []
        for i in range(self.validationMatrix.shape[0]):
            for j in range(self.validationMatrix.shape[1]):
                if self.validationMatrix[i, j] == 1:
                    all_ones.append((i, j))
        
        self.measureClusters = []
        
        # 2. 进行聚类
        row_is_scanned = [False] * self.validationMatrix.shape[0]
        col_is_scanned = [False] * self.validationMatrix.shape[1]
        completed_ones = [False] * len(all_ones)
        
        for i in range(self.validationMatrix.shape[0]):
            if not row_is_scanned[i]:
                row_is_scanned[i] = True
                is_new_point = True
                
                while is_new_point:
                    is_new_point = False
                    for j, (row, col) in enumerate(all_ones):
                        if row_is_scanned[row] and not col_is_scanned[col]:
                            col_is_scanned[col] = True
                            is_new_point = True
                        elif col_is_scanned[col] and not row_is_scanned[row]:
                            row_is_scanned[row] = True
                            is_new_point = True
                
                cluster_points = []
                for k, (row, col) in enumerate(all_ones):
                    if row_is_scanned[row] and col_is_scanned[col] and not completed_ones[k]:
                        cluster_points.append((row, col))
                        completed_ones[k] = True
                
                if cluster_points:
                    self.measureClusters.append(cluster_points)
    
    def recur_enum(self, cluster):
        """递归枚举所有可能的关联事件"""
        if not cluster or len(cluster) > 15:
            return [[]]
        
        all_feasible_event = []
        
        # 枚举第一个1
        for i in range(len(cluster)):
            sub_cluster = []
            for j in range(i, len(cluster)):
                if cluster[j][0] != cluster[i][0] and cluster[j][1] != cluster[i][1]:
                    sub_cluster.append(cluster[j])
            
            # 递归处理子聚类
            sub_all_feasible_event = self.recur_enum(sub_cluster)
            
            # 将第一个1加入所有可行的子事件
            for event in sub_all_feasible_event:
                new_event = event.copy()
                new_event.append(cluster[i])
                all_feasible_event.append(new_event)
        
        # 没有1存在于聚类中
        all_feasible_event.append([])
        
        return all_feasible_event
    
    def calc_joint_prob(self, one_cluster, track_index_vec, measure_index_vec, prob_vec):
        """计算联合概率
        
        Args:
            one_cluster: 一个聚类
            track_index_vec: 轨迹索引向量
            measure_index_vec: 测量索引向量
            prob_vec: 概率向量
        """
        # 获取轨迹和测量索引
        for point in one_cluster:
            is_new_track = True
            for track_idx in track_index_vec:
                if point[0] == track_idx:
                    is_new_track = False
                    break
            if is_new_track:
                track_index_vec.append(point[0])
        
        measure_count_vec = []
        for point in one_cluster:
            is_new_measure = True
            for measure_idx in measure_count_vec:
                if point[1] == measure_idx:
                    is_new_measure = False
                    break
            if is_new_measure:
                measure_count_vec.append(point[1])
        
        n_tracks = len(track_index_vec)
        n_measures = len(measure_count_vec)
        
        # 1. 枚举所有可行的关联事件
        all_feasible_event = self.recur_enum(one_cluster)
        if all_feasible_event and all_feasible_event[-1] == []:
            all_feasible_event.pop()
        
        # 2. 计算联合概率
        total_volume = 0.0
        for track_idx in track_index_vec:
            total_volume += self.tracks[track_idx].gatingVolume
        
        Pr = []
        for event in all_feasible_event:
            N = 1.0
            a1 = 1
            a_detect = 1.0
            
            for point in event:
                # 测量概率，第一项遵循高斯分布
                i_track = point[0]
                i_measure = point[1]
                temp_dist = self.distanceMatrix[i_track, i_measure]
                N = N * np.exp(-0.5 * temp_dist * temp_dist) / np.sqrt(2.0 * np.pi * self.tracks[i_track].detS)
                
                # 检测概率，第二项
                for nn in range(n_tracks):
                    target_indicator = 0
                    if i_track == track_index_vec[nn]:
                        target_indicator = 1
                    a_detect *= (self.Pd ** target_indicator) * ((1 - self.Pd) ** (1 - target_indicator))
            
            i_false_num = n_measures - len(event)
            for kk in range(1, i_false_num + 1):
                a1 *= kk
            
            Pr.append(N * a_detect * a1 / (total_volume ** i_false_num))
        
        # 归一化
        Pr_sum = sum(Pr)
        if Pr_sum > 0:
            Pr = [p / Pr_sum for p in Pr]
        
        # 3. 结合所有可行事件以获得最终概率
        for i, track_idx in enumerate(track_index_vec):
            one_measure_vec = []
            
            for point in one_cluster:
                if point[0] == track_idx:
                    is_new_measure = True
                    for measure_idx in one_measure_vec:
                        if point[1] == measure_idx:
                            is_new_measure = False
                            break
                    if is_new_measure:
                        one_measure_vec.append(point[1])
            
            measure_index_vec.append(one_measure_vec)
            
            one_prob_vec = [0.0] * len(one_measure_vec)
            for j, event in enumerate(all_feasible_event):
                for point in event:
                    if point[0] == track_idx:
                        for n, measure_idx in enumerate(one_measure_vec):
                            if point[1] == measure_idx:
                                one_prob_vec[n] += Pr[j]
            
            prob_vec.append(one_prob_vec)
    
    def correct_each_cluster(self, all_measurements):
        """修正每个聚类的轨迹状态"""
        # 修正有有效测量的轨迹
        for i, cluster in enumerate(self.measureClusters):
            track_index_vec = []
            measure_index_vec = []
            prob_vec = []
            
            self.calc_joint_prob(cluster, track_index_vec, measure_index_vec, prob_vec)
            
            for j, track_idx in enumerate(track_index_vec):
                track_valid_mea_vec = []
                for measure_idx in measure_index_vec[j]:
                    track_valid_mea_vec.append(all_measurements[measure_idx])
                
                self.tracks[track_idx].correct(track_valid_mea_vec, prob_vec[j])
        
        # 修正没有有效测量的轨迹
        if self.validationMatrix.shape[0] > 0 and self.validationMatrix.shape[1] > 0:
            row_sum = np.sum(self.validationMatrix, axis=1)
            for i, sum_val in enumerate(row_sum):
                if sum_val == 0:
                    self.tracks[i].correct([], [])
    
    def delete_tracks(self):
        """删除不活跃的轨迹"""
        cov_mat = np.eye(2, dtype=np.float32) * (self.processNoise + self.measureNoise)
        eigenvalues, _ = np.linalg.eigh(cov_mat)
        major_axis = 2.0 * self.GateThreshold * np.sqrt(eigenvalues[0])
        
        terminate_threshold = major_axis * 2.0
        
        i = 0
        while i < len(self.tracks):
            track = self.tracks[i]
            if (track.gatingEllipse[1][0] >= terminate_threshold and 
                track.gatingEllipse[1][1] >= terminate_threshold) or track.consecutive_misses > 10:
                self.tracks.pop(i)
            else:
                i += 1
    
    def init_unassociate_tracks(self):
        """初始化未关联的轨迹"""
        for measure in self.unValidatedMeasures:
            self.create_new_track(measure)
    
    def create_new_track(self, init_pt):
        """创建新轨迹"""
        self.trackID += 1
        track = JPDATrack(init_pt, self.processNoise, self.measureNoise, 
                          self.Pd, self.Pg, self.trackID)
        self.tracks.append(track)
        
    def get_active_tracks(self):
        """获取活跃的轨迹"""
        return [track for track in self.tracks if track.age >= 5]