# SLAM + IMU(BNO055, 방식 B) + 즉각회피(C/L/R) + RECOVER + 카메라 박스 통합
# 회피 우선순위:
#   1차) 초음파 C/L/R 또는 LiDAR 정면 또는 카메라 → 디바운스 없이 즉시 회피
#   2차) 즉각회피가 계속 실패하면 RECOVER(후진 탈출)로 승격
# 회전각(theta) = 자이로 92% + 엔코더 8% / 위치(x,y) = 엔코더 적분

import time
import math
import heapq
import serial
import board
import busio
import adafruit_mlx90640
import cv2
import numpy as np
import glob
import subprocess
import shutil
from gpiozero import PWMOutputDevice, OutputDevice, RotaryEncoder
from pyrplidar import PyRPlidar

# ==========================================
# 0. Global tuning constants
# ==========================================
WHEEL_BASE          = 0.375
LIDAR_ANGLE_SIGN    = 1.0
MAP_SIZE_M          = 10.0
MAP_RES             = 0.05
REPORT_INTERVAL     = 2.0
REPLAN_INTERVAL     = 5.0
WAYPOINT_TIMEOUT    = 15.0
MAP_UPDATE_INTERVAL = 0.4
FIRE_DETECT_TEMP    = 60.0
FIRE_ARRIVE_TEMP    = 120.0
GYRO_TRUST          = 0.92      # 상보필터: 자이로 신뢰 비율

# --- 즉각 회피 튜닝 ---
US_NEAR_TH          = 550       # [mm] 초음파 회피 임계 (너무 예민하면 450, 둔하면 650)
US_VALID_MIN        = 150       # [mm] 초음파 유효 최소 (이하 노이즈 무시)
US_VALID_MAX        = 2000      # [mm] 초음파 유효 최대 (이상 헛값 무시: R의 9990 등)
LIDAR_FRONT_TH      = 500       # [mm] LiDAR 정면 회피 임계
IMM_AVOID_LIMIT     = 12        # 즉각회피 N프레임 지속되면 RECOVER로 승격

# ==========================================
# 1. Hardware Class Definitions
# ==========================================

class RobotDrive:
    def __init__(self):
        self.m1_pwm = PWMOutputDevice(12)
        self.m1_dir = OutputDevice(5)
        self.m1_enc = RotaryEncoder(17, 27, max_steps=1000000)

        self.m2_pwm = PWMOutputDevice(13)
        self.m2_dir = OutputDevice(6)
        self.m2_enc = RotaryEncoder(22, 24, max_steps=1000000)

        self.PULSES_PER_REV = 400
        self.WHEEL_DIAMETER = 0.0556
        self.WHEEL_CIRCUMFERENCE = 3.14159 * self.WHEEL_DIAMETER
        self.last_steps = (0, 0)

    def set_speed(self, m1_Tmot, m2_Tmot):
        self.m1_dir.value = m1_Tmot > 0
        self.m1_pwm.value = min(abs(m1_Tmot), 1.0)
        self.m2_dir.value = m2_Tmot < 0
        self.m2_pwm.value = min(abs(m2_Tmot), 1.0)

    def stop(self):
        self.set_speed(0.0, 0.0)

    def get_wheel_travel(self):
        s1 = self.m1_enc.steps
        s2 = self.m2_enc.steps
        ds1 = s1 - self.last_steps[0]
        ds2 = s2 - self.last_steps[1]
        self.last_steps = (s1, s2)

        d1 = ds1 / self.PULSES_PER_REV * self.WHEEL_CIRCUMFERENCE
        d2 = -ds2 / self.PULSES_PER_REV * self.WHEEL_CIRCUMFERENCE

        MAX_TRAVEL = 0.05
        d1 = max(min(d1, MAX_TRAVEL), -MAX_TRAVEL)
        d2 = max(min(d2, MAX_TRAVEL), -MAX_TRAVEL)

        return d1, d2

class LinearActuator:
    def __init__(self):
        self.act_r_pwm = PWMOutputDevice(18, frequency=1000)
        self.act_l_pwm = PWMOutputDevice(23, frequency=1000)

    def move(self, action, Tmot):
        safe_Tmot = max(min(Tmot, 1.0), 0.0)
        if action == 'extend':
            self.act_l_pwm.value = 0.0
            self.act_r_pwm.value = safe_Tmot
        elif action == 'retract':
            self.act_r_pwm.value = 0.0
            self.act_l_pwm.value = safe_Tmot

    def stop(self):
        self.act_r_pwm.value = 0.0
        self.act_l_pwm.value = 0.0

class CombineMotor:
    def __init__(self):
        self.comb_r_pwm = PWMOutputDevice(14, frequency=8000)
        self.comb_l_pwm = PWMOutputDevice(15, frequency=8000)

    def run(self, action, Tmot):
        safe_Tmot = max(min(Tmot, 1.0), 0.0)
        if action == 'suction':
            self.comb_l_pwm.value = 0.0
            self.comb_r_pwm.value = safe_Tmot
        elif action == 'discharge':
            self.comb_r_pwm.value = 0.0
            self.comb_l_pwm.value = safe_Tmot

    def stop(self):
        self.comb_r_pwm.value = 0.0
        self.comb_l_pwm.value = 0.0

class ArduinoController:
    """Handles Water Pump and Center/Left/Right Ultrasonic Sensors"""
    def __init__(self):
        PORT = '/dev/ttyACM0'
        BAUD_RATE = 115200
        self.center_us = 9999.0
        self.left_us = 9999.0
        self.right_us = 9999.0

        try:
            self.py_serial = serial.Serial(PORT, BAUD_RATE, timeout=0.1)
            time.sleep(2)
            self.is_connected = True
            print("[INFO] Arduino connected (Pump + 3x Ultrasonic).")
        except serial.SerialException:
            print("[WARNING] Arduino serial connection failed")
            self.is_connected = False

    def update(self):
        if not self.is_connected: return
        try:
            while self.py_serial.in_waiting > 0:
                line = self.py_serial.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("U:"):
                    parts = line[2:].split(',')
                    if len(parts) >= 3:
                        self.center_us = float(parts[0]) * 10.0
                        self.left_us = float(parts[1]) * 10.0
                        self.right_us = float(parts[2]) * 10.0
                    elif len(parts) == 2:
                        self.left_us = float(parts[0]) * 10.0
                        self.right_us = float(parts[1]) * 10.0
        except Exception:
            pass

    def start_pump(self):
        if self.is_connected: self.py_serial.write(b'S')

    def stop_pump(self):
        if self.is_connected: self.py_serial.write(b'O')

class ThermalCamera:
    def __init__(self):
        self.is_connected = False
        i2c = busio.I2C(board.SCL, board.SDA)
        for attempt in range(3):
            try:
                self.mlx = adafruit_mlx90640.MLX90640(i2c)
                self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
                self.frame = [0] * 768
                self.is_connected = True
                print("[INFO] Thermal Camera connected.")
                break
            except Exception:
                time.sleep(1)
        if not self.is_connected: print("[WARNING] Thermal Camera failed.")

    def update(self):
        if self.is_connected:
            try: self.mlx.getFrame(self.frame)
            except Exception: pass

    def get_fire_data(self):
        if not self.is_connected: return 0.0, 15
        data_array = np.array(self.frame).reshape((24, 32))
        temp_max = np.max(data_array)
        max_idx = np.unravel_index(np.argmax(data_array, axis=None), data_array.shape)
        return temp_max, max_idx[1]

    def get_thermal_image(self):
        if not self.is_connected:
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(img, "THERMAL: NO SIGNAL", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return img
        data_array = np.array(self.frame).reshape((24, 32))
        clipped = np.clip(data_array, 20.0, 100.0)
        norm = np.uint8((clipped - 20.0) / 80.0 * 255)
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (320, 240), interpolation=cv2.INTER_CUBIC)
        cv2.putText(heatmap, f"Max: {np.max(data_array):.1f} C", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return heatmap

class RGBCamera:
    def __init__(self):
        self.current_frame = None
        self.is_connected = False
        self.cap = None
        self.obstacle_boxes = []
        video_nodes = sorted(glob.glob('/dev/video*'), key=lambda p: int(p.replace('/dev/video', '')))
        for node in video_nodes:
            idx = int(node.replace('/dev/video', ''))
            try:
                with open(f'/sys/class/video4linux/video{idx}/name', 'r') as f:
                    name = f.read().lower()
                    if 'codec' in name or 'isp' in name or 'rpivid' in name or 'unicam' in name: continue
            except Exception: pass

            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, test = cap.read()
                if ret:
                    self.cap = cap
                    self.is_connected = True
                    print(f"[INFO] Webcam connected on /dev/video{idx}")
                    break
            cap.release()

    def update(self):
        if not self.is_connected: return
        ret, frame = self.cap.read()
        if ret: self.current_frame = cv2.resize(frame, (320, 240))

    def detect_obstacle(self):
        # 화면 전체(위 15% 제외)에서 장애물 영역을 박스로 추출.
        # 주의: 반환값은 표시(시각화) 용도. 실제 회피 판단에는 사용하지 않는다(옵션 A).
        if not self.is_connected or self.current_frame is None:
            self.obstacle_boxes = []
            return False
        h, w, _ = self.current_frame.shape
        roi_y = int(h * 0.15)                       # 위쪽 15%(천장/조명) 제외, 나머지 전체
        roi = self.current_frame[roi_y:h, 0:w]
        edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150)
        ratio = cv2.countNonZero(edges) / (edges.shape[0] * edges.shape[1])

        self.obstacle_boxes = []
        if ratio > 0.035:
            kernel = np.ones((9, 9), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw * bh > 1500:                  # 큰 덩어리만 (자잘한 노이즈 박스 방지)
                    self.obstacle_boxes.append((x, y + roi_y, bw, bh))
        return ratio > 0.035

    def get_display_frame(self, state_text=""):
        frame = self.current_frame.copy() if self.current_frame is not None else np.zeros((240, 320, 3), dtype=np.uint8)
        boxes = getattr(self, 'obstacle_boxes', [])

        for (x, y, bw, bh) in boxes:
            # 박스 크기로 위험도 색 결정 (클수록=가까울수록 빨강)
            area_ratio = (bw * bh) / float(frame.shape[0] * frame.shape[1])
            if area_ratio > 0.15:
                color, risk = (0, 0, 255), "DANGER"      # 빨강
            elif area_ratio > 0.06:
                color, risk = (0, 165, 255), "WARNING"   # 주황
            else:
                color, risk = (0, 255, 0), "DETECTED"    # 초록

            # 모서리 브래킷 스타일 (AI 감시 느낌)
            L = max(12, min(bw, bh) // 4)
            x2, y2 = x + bw, y + bh
            for (cx, cy, dx, dy) in [(x, y, 1, 1), (x2, y, -1, 1), (x, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, 2)
                cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, 2)

            # 라벨 배경 + 글자
            (tw, th), _ = cv2.getTextSize(risk, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            ly = max(y - 4, th + 4)
            cv2.rectangle(frame, (x, ly - th - 4), (x + tw + 6, ly + 2), color, -1)
            cv2.putText(frame, risk, (x + 3, ly - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # 상단 감지 카운터 (HUD 느낌)
        if boxes:
            hud = f"OBJECTS DETECTED: {len(boxes)}"
            (hw, hh), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (5, 5), (15 + hw, 22), (0, 0, 0), -1)
            cv2.putText(frame, hud, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if state_text:
            cv2.putText(frame, f"STATE: {state_text}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def stop(self):
        if self.cap: self.cap.release()

class IMU:
    """BNO055 — 방식 B에서는 Z축 자이로 각속도(gyro_z)만 사용 (i2c-3)"""
    def __init__(self, bus_num=3):
        from adafruit_extended_bus import ExtendedI2C as I2C
        import adafruit_bno055
        self.is_connected = False
        try:
            i2c = I2C(bus_num)
            self.sensor = adafruit_bno055.BNO055_I2C(i2c)
            time.sleep(0.5)
            self.is_connected = True
            print(f"[INFO] IMU (BNO055) connected on i2c-{bus_num}.")
        except Exception as e:
            print(f"[WARNING] IMU connection failed: {e}")

    def get_gyro_z(self):
        """Z축 각속도 [rad/s]. 실패 시 None. (deg/s로 나오면 radians 변환 줄로 교체)"""
        if not self.is_connected:
            return None
        try:
            g = self.sensor.gyro
            if g is None or g[2] is None:
                return None
            return g[2]
            # return math.radians(g[2])   # ← 자이로가 deg/s로 나오면 이 줄로 교체
        except Exception:
            return None

    def get_calib(self):
        if not self.is_connected:
            return (0, 0, 0, 0)
        try:
            c = self.sensor.calibration_status
            return c if c else (0, 0, 0, 0)
        except Exception:
            return (0, 0, 0, 0)

# ==========================================
# 2. Pure-Python SLAM
# ==========================================
class OccupancyGridSLAM:
    L_OCC, L_FREE, L_CLAMP = 1.1, -0.35, 8.0
    OCC_TH, FREE_TH = 1.5, -1.0

    def __init__(self):
        self.size = int(MAP_SIZE_M / MAP_RES)
        self.log_odds = np.zeros((self.size, self.size), dtype=np.float32)
        self.x, self.y, self.theta = MAP_SIZE_M / 2.0, MAP_SIZE_M / 2.0, 0.0

    def update_odometry(self, d_left, d_right, gyro_z=None, dt=0.0):
        d = (d_left + d_right) / 2.0
        dtheta_enc = (d_right - d_left) / WHEEL_BASE
        if gyro_z is not None and dt > 0:
            dtheta_imu = gyro_z * dt
            dtheta = GYRO_TRUST * dtheta_imu + (1.0 - GYRO_TRUST) * dtheta_enc
        else:
            dtheta = dtheta_enc
        self.theta = (self.theta + dtheta + math.pi) % (2 * math.pi) - math.pi
        self.x += d * math.cos(self.theta)
        self.y += d * math.sin(self.theta)

    def pose_cell(self):
        return (max(0, min(self.size - 1, int(self.x / MAP_RES))), max(0, min(self.size - 1, int(self.y / MAP_RES))))

    def update_map(self, scan):
        rx, ry = self.pose_cell()
        for ang_deg, dist_mm in scan:
            if not (250 < dist_mm < 6000): continue
            d = dist_mm / 1000.0
            bearing = self.theta + LIDAR_ANGLE_SIGN * math.radians(ang_deg)
            gx, gy = max(0, min(self.size - 1, int((self.x + d * math.cos(bearing)) / MAP_RES))), max(0, min(self.size - 1, int((self.y + d * math.sin(bearing)) / MAP_RES)))

            x0, y0, x1, y1 = rx, ry, gx, gy
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
            err = dx - dy
            cells = []
            while True:
                cells.append((x0, y0))
                if x0 == x1 and y0 == y1: break
                e2 = 2 * err
                if e2 > -dy: err -= dy; x0 += sx
                if e2 < dx: err += dx; y0 += sy

            for (cx, cy) in cells[:-1]: self.log_odds[cy, cx] = max(self.log_odds[cy, cx] + self.L_FREE, -self.L_CLAMP)
            self.log_odds[gy, gx] = min(self.log_odds[gy, gx] + self.L_OCC, self.L_CLAMP)

    def mark_virtual_obstacle(self, dist_m=0.30):
        gx, gy = max(0, min(self.size - 1, int((self.x + dist_m * math.cos(self.theta)) / MAP_RES))), max(0, min(self.size - 1, int((self.y + dist_m * math.sin(self.theta)) / MAP_RES)))
        self.log_odds[max(0, gy - 1):gy + 2, max(0, gx - 1):gx + 2] = self.L_CLAMP

    def occupied_mask(self): return self.log_odds > self.OCC_TH
    def free_mask(self): return self.log_odds < self.FREE_TH
    def unknown_mask(self): return np.abs(self.log_odds) <= 0.05

    def get_map_image(self, path=None, waypoint=None):
        img = np.full((self.size, self.size), 128, dtype=np.uint8)
        img[self.free_mask()] = 255
        img[self.occupied_mask()] = 0
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if path:
            for (px, py) in path: img[py, px] = (255, 0, 0)
        if waypoint: cv2.circle(img, waypoint, 3, (0, 165, 255), -1)
        rx, ry = self.pose_cell()
        cv2.circle(img, (rx, ry), 3, (0, 0, 255), -1)
        cv2.line(img, (rx, ry), (int(rx + 8 * math.cos(self.theta)), int(ry + 8 * math.sin(self.theta))), (0, 255, 0), 1)
        return cv2.flip(cv2.resize(img, (600, 600), interpolation=cv2.INTER_NEAREST), 0)

    def correct_pose_with_lidar(self, scan):
        if np.count_nonzero(self.occupied_mask()) < 80:
            return
        valid = [(a, d) for a, d in scan if 150 < d < 6000]
        if len(valid) < 20:
            return
        angles = np.radians([s[0] for s in valid]) * LIDAR_ANGLE_SIGN
        distances = np.array([s[1] for s in valid]) / 1000.0

        def score_at(offset):
            bearings = self.theta + offset + angles
            gx = ((self.x + distances * np.cos(bearings)) / MAP_RES).astype(int)
            gy = ((self.y + distances * np.sin(bearings)) / MAP_RES).astype(int)
            m = (gx >= 0) & (gx < self.size) & (gy >= 0) & (gy < self.size)
            return np.sum(self.log_odds[gy[m], gx[m]]) if np.any(m) else -1e9

        base = score_at(0.0)
        best_off, best_score = 0.0, base
        for offset in np.linspace(-math.radians(6), math.radians(6), 7):
            s = score_at(offset)
            if s > best_score:
                best_score, best_off = s, offset
        if best_off != 0.0 and best_score > base * 1.05:
            self.theta += best_off

# ==========================================
# 3. Global Planner
# ==========================================
class PathPlanner:
    INFLATE_CELLS = 6

    def __init__(self, slam):
        self.slam = slam

    def _inflated_obstacles(self):
        return cv2.dilate(self.slam.occupied_mask().astype(np.uint8), np.ones((2 * self.INFLATE_CELLS + 1,) * 2, np.uint8)) > 0

    def find_frontier_waypoint(self):
        frontier = self.slam.free_mask() & (cv2.dilate(self.slam.unknown_mask().astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~self._inflated_obstacles()
        ys, xs = np.nonzero(frontier)
        if len(xs) == 0: return None
        rx, ry = self.slam.pose_cell()
        d2 = (xs - rx) ** 2 + (ys - ry) ** 2
        valid = d2 > int(0.6 / MAP_RES) ** 2
        if not np.any(valid): return None
        i = np.argmin(np.where(valid, d2 + 5000 * np.abs((np.arctan2(ys - ry, xs - rx) - self.slam.theta + np.pi) % (2 * np.pi) - np.pi), np.inf))
        return (int(xs[i]), int(ys[i]))

    def astar(self, start, goal):
        blocked = self._inflated_obstacles()
        if blocked[goal[1], goal[0]]: return None
        moves = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414)]
        g, came, open_heap, closed = {start: 0.0}, {}, [(0.0, start)], set()
        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == goal:
                path = [cur]
                while cur in came: cur = came[cur]; path.append(cur)
                return path[::-1][::3]
            if cur in closed: continue
            closed.add(cur)
            for dx, dy, cost in moves:
                nxt = (cur[0] + dx, cur[1] + dy)
                if 0 <= nxt[0] < self.slam.size and 0 <= nxt[1] < self.slam.size and not blocked[nxt[1], nxt[0]]:
                    ng = g[cur] + cost
                    if ng < g.get(nxt, float('inf')):
                        g[nxt], came[nxt] = ng, cur
                        heapq.heappush(open_heap, (ng + math.hypot(goal[0] - nxt[0], goal[1] - nxt[1]), nxt))
        return None

# ==========================================
# 4. Local Navigator (히스토그램 + 평활화)
# ==========================================
class LocalNavigator:
    NUM_SECTORS = 36
    BASE_SPEED = 0.25
    TURN_GAIN = 0.012
    DEADBAND_DEG = 8.0
    HOLD_TIME = 2.5

    def __init__(self):
        self.hist = np.full(self.NUM_SECTORS, 9999.0)
        self.stamp = np.zeros(self.NUM_SECTORS)
        self.mode = 'CRUISE'

    def update_histogram(self, scan, now):
        fresh = np.full(self.NUM_SECTORS, np.inf)
        n_valid = 0
        for ang_deg, dist_mm in scan:
            if dist_mm > 150:
                idx = int((ang_deg % 360) / 10) % self.NUM_SECTORS
                fresh[idx] = min(fresh[idx], dist_mm)
                n_valid += 1
        if n_valid < 30:
            return
        for i in range(self.NUM_SECTORS):
            if np.isfinite(fresh[i]):
                if self.hist[i] >= 9999.0:
                    self.hist[i] = fresh[i]
                else:
                    self.hist[i] = 0.6 * fresh[i] + 0.4 * self.hist[i]
                self.stamp[i] = now
            elif now - self.stamp[i] > self.HOLD_TIME:
                self.hist[i] = 9999.0

    def sector_min(self, deg_from, deg_to):
        return min(self.hist[i] for i in {int((d % 360) / 10) % self.NUM_SECTORS for d in range(deg_from, deg_to, 5)})

# ==========================================
# 5. FSM Supervisor
# ==========================================
class FireFighterFSM:
    def __init__(self):
        print("=== Initializing Robot Hardware ===")
        self.drive = RobotDrive()
        self.actuator = LinearActuator()
        self.combine = CombineMotor()
        self.arduino = ArduinoController()
        self.thermal = ThermalCamera()
        self.rgb_cam = RGBCamera()
        self.imu = IMU(bus_num=3)
        self.target_heading = None
        self._last_odo_t = time.time()

        ports = glob.glob('/dev/ttyUSB*')
        self.lidar = PyRPlidar()
        if ports:
            self.lidar.connect(port=ports[0], baudrate=460800, timeout=3)
            try:
                self.lidar.stop()
                self.lidar.set_motor_pwm(0)
                time.sleep(1)
            except Exception: pass
            self.lidar.set_motor_pwm(500)
            time.sleep(2)
            self.scan_iterator = self.lidar.start_scan()()
        else:
            print("[WARNING] LiDAR connection failed.")
            self.scan_iterator = None

        self.slam = OccupancyGridSLAM()
        self.planner = PathPlanner(self.slam)
        self.local_nav = LocalNavigator()
        self.path, self.waypoint = [], None
        self.state, self.state_start_time = 'INIT', time.time()
        self.init_flag = False
        self.last_print, self.last_report = time.time(), time.time()
        self.latest_scan = []
        self.waypoint_set_time, self.last_map_update = 0.0, 0.0
        self.last_replan = 0.0
        self.last_decision = ""

        # 즉각회피 / RECOVER 상태 변수
        self.imm_avoid = 0
        self.recover_phase    = 'BACK'
        self.recover_t0       = 0.0
        self.recover_turn_dir = 'right'
        self.recover_attempts = 0
        self.recover_enter_t  = 0.0

        cv2.namedWindow("RGB Camera", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Thermal Camera", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("SLAM Map", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("RGB Camera", 0, 0)
        cv2.moveWindow("Thermal Camera", 340, 0)
        cv2.moveWindow("SLAM Map", 680, 0)

        if self.imu.is_connected:
            print("[INFO] 방식 B(자이로 융합). 시작 시 로봇을 10초 이상 가만히 두면 gyro 캘리브레이션(=3)이 됩니다.")

    def _collect_scan(self):
        scan = []
        if not self.scan_iterator: return scan
        try:
            for _ in range(300):
                s = next(self.scan_iterator)
                ang = s.angle - 360 if s.angle > 180 else s.angle
                scan.append((ang, s.distance))
        except StopIteration: pass
        return scan

    def _us_valid(self, v):
        return US_VALID_MIN < v < US_VALID_MAX

    def _sensor_report(self):
        front = self.local_nav.sector_min(-30, 31)
        left = self.local_nav.sector_min(-90, -25)
        right = self.local_nav.sector_min(25, 91)
        cam_obs = self.rgb_cam.detect_obstacle()
        temp_max, _ = self.thermal.get_fire_data()
        cal = self.imu.get_calib()
        print(f"[SENSOR REPORT] LiDAR(mm) L:{left:.0f} F:{front:.0f} R:{right:.0f} | "
              f"US(mm) C:{self.arduino.center_us:.0f} L:{self.arduino.left_us:.0f} R:{self.arduino.right_us:.0f} | "
              f"CamObstacle:{cam_obs} | NavMode:{self.local_nav.mode} | "
              f"MaxTemp:{temp_max:.1f}C | IMU_cal(s/g/a/m):{cal[0]}/{cal[1]}/{cal[2]}/{cal[3]} | "
              f"Pose:({self.slam.x:.2f},{self.slam.y:.2f},{math.degrees(self.slam.theta):.0f}deg)")

    def run(self):
        print("\n=== FSM Control Started ===")
        try:
            while True:
                self.rgb_cam.update()
                self.thermal.update()
                self.arduino.update()
                self.latest_scan = self._collect_scan()

                now = time.time()
                dt = now - self._last_odo_t
                self._last_odo_t = now

                gz = self.imu.get_gyro_z()
                dl, dr = self.drive.get_wheel_travel()
                self.slam.update_odometry(dl, dr, gyro_z=gz, dt=dt)

                if self.latest_scan and now - self.last_map_update > MAP_UPDATE_INTERVAL:
                    self.slam.correct_pose_with_lidar(self.latest_scan[::2])
                    self.slam.update_map(self.latest_scan[::2])
                    self.last_map_update = now

                cv2.imshow("RGB Camera", self.rgb_cam.get_display_frame(self.state))
                cv2.imshow("Thermal Camera", self.thermal.get_thermal_image())
                cv2.imshow("SLAM Map", self.slam.get_map_image(self.path, self.waypoint))
                if cv2.waitKey(1) & 0xFF == ord('q'): break

                if now - self.last_print >= 1.0:
                    print(f"\n[Execution Monitor] State: {self.state}")
                    self.last_print = now
                if now - self.last_report >= REPORT_INTERVAL:
                    self._sensor_report()
                    self.last_report = now

                if self.state == 'INIT': self._state_init()
                elif self.state == 'PATROL': self._state_patrol(now)
                elif self.state == 'RECOVER': self._state_recover(now)
                elif self.state == 'APPROACH': self._state_approach()
                elif self.state == 'CLOSE_IN': self._state_close_in()
                elif self.state == 'DEPLOY': self._state_deploy()
                elif self.state == 'EXTINGUISH': self._state_extinguish()
                elif self.state == 'RESET': self._state_reset()

                time.sleep(0.01)
        except KeyboardInterrupt:
            self._emergency_stop()
        finally:
            self._emergency_stop()

    def _state_init(self):
        if not self.init_flag:
            print("\n[INIT] Pulling combine up...")
            self.actuator.move('retract', 1.0)
            self.state_start_time = time.time()
            self.init_flag = True
        if time.time() - self.state_start_time >= 6.0:
            self.actuator.stop()
            self.state = 'PATROL'

    def _state_patrol(self, now):
        # --- 0. 불씨 감지 (연속 3프레임) ---
        temp_max, _ = self.thermal.get_fire_data()
        if temp_max > FIRE_DETECT_TEMP:
            self.fire_count = getattr(self, 'fire_count', 0) + 1
        else:
            self.fire_count = 0
        if self.fire_count >= 3:
            self.fire_count = 0
            print(f"\n[Ember detected!] Temp: {temp_max:.1f}C -> APPROACH")
            self.drive.stop()
            self.path, self.state = [], 'APPROACH'
            return

        self.local_nav.update_histogram(self.latest_scan, now)
        front   = self.local_nav.sector_min(-20, 21)
        front_w = self.local_nav.sector_min(-40, 41)
        left    = self.local_nav.sector_min(-90, -25)
        right   = self.local_nav.sector_min(25, 91)
        # 카메라는 표시(초록 박스)용으로만 호출. 회피 판단에는 쓰지 않는다(옵션 A).
        self.rgb_cam.detect_obstacle()

        # 초음파 유효값만 (R 헛값 9990·120 등 필터)
        c_ok = self.arduino.center_us if self._us_valid(self.arduino.center_us) else 9999
        l_ok = self.arduino.left_us   if self._us_valid(self.arduino.left_us)   else 9999
        r_ok = self.arduino.right_us  if self._us_valid(self.arduino.right_us)  else 9999

        # === 1차: 즉각 회피 (디바운스 없음) — LiDAR/초음파만으로 판단 ===
        if (c_ok < US_NEAR_TH or l_ok < US_NEAR_TH or r_ok < US_NEAR_TH
                or front < LIDAR_FRONT_TH):
            self.drive.stop()
            # 더 트인 쪽으로 즉시 제자리 회전
            if left > right:
                self.drive.set_speed(-0.40, 0.40)
            else:
                self.drive.set_speed(0.40, -0.40)
            # 계속 못 빠져나가면 RECOVER로 승격
            self.imm_avoid += 1
            if self.imm_avoid >= IMM_AVOID_LIMIT:
                self.imm_avoid = 0
                self.recover_turn_dir = 'left' if left > right else 'right'
                self.recover_phase, self.recover_t0 = 'BACK', now
                self.recover_attempts = 0
                self.recover_enter_t = now
                self.path, self.waypoint = [], None
                self.state = 'RECOVER'
                print(f"   -> [RECOVER] 즉각회피 실패 → 후진 탈출 (F:{front:.0f} C:{c_ok:.0f} L:{l_ok:.0f} R:{r_ok:.0f}), dir={self.recover_turn_dir}")
            return
        else:
            self.imm_avoid = 0

        # === 2차: 평상 주행 (거리 비례 속도) ===
        if front_w > 1000:
            self.drive.set_speed(0.30, 0.30)
        elif front_w > 500:
            speed = 0.15 + 0.15 * (front_w - 500) / 500.0
            if left > right:
                self.drive.set_speed(speed * 0.5, speed)
            else:
                self.drive.set_speed(speed, speed * 0.5)
        else:
            if left > right:
                self.drive.set_speed(-0.25, 0.25)
            else:
                self.drive.set_speed(0.25, -0.25)
        self.path, self.waypoint = [], None
        return

    def _state_recover(self, now):
        # 불씨 최우선
        temp_max, _ = self.thermal.get_fire_data()
        if temp_max > FIRE_DETECT_TEMP:
            self.drive.stop()
            self.path, self.state = [], 'APPROACH'
            return

        # 안전장치: 15초 이상 RECOVER면 강제 복귀
        if now - self.recover_enter_t > 15.0:
            print("   -> [RECOVER] 15초 초과 → 강제 PATROL 복귀")
            self.drive.stop()
            self.state = 'PATROL'
            return

        self.local_nav.update_histogram(self.latest_scan, now)
        front = self.local_nav.sector_min(-20, 21)
        c_us  = self.arduino.center_us
        # 탈출 판정은 LiDAR 위주. 초음파는 250mm 이내일 때만 반영(헛값 무시)
        front_block = front
        if 150 < c_us < 250:
            front_block = min(front, c_us)

        BACK_TIME    = 0.8
        TURN_TIMEOUT = 3.0
        CLEAR_TH     = 1000
        MAX_ATTEMPTS = 3

        if self.recover_phase == 'BACK':
            if now - self.recover_t0 < BACK_TIME:
                self.drive.set_speed(-0.25, -0.25)
                return
            self.drive.stop()
            self.recover_phase = 'TURN'
            self.recover_t0 = now
            return

        if self.recover_phase == 'TURN':
            if front_block > CLEAR_TH:
                self.drive.stop()
                self.last_replan = 0.0
                self.state = 'PATROL'
                print(f"   -> [RECOVER] 전방 확보 (F:{front_block:.0f}) → PATROL 복귀")
                return
            if now - self.recover_t0 > TURN_TIMEOUT:
                self.recover_attempts += 1
                if self.recover_attempts >= MAX_ATTEMPTS:
                    print("   -> [RECOVER][경고] 후진·회전 반복 실패. 전방 LiDAR 자기-가림 가능성.")
                    self.recover_attempts = 0
                self.recover_turn_dir = 'left' if self.recover_turn_dir == 'right' else 'right'
                self.recover_phase = 'BACK'
                self.recover_t0 = now
                return
            if self.recover_turn_dir == 'left':
                self.drive.set_speed(-0.40, 0.40)
            else:
                self.drive.set_speed(0.40, -0.40)
            return

    def _state_approach(self):
        temp_max, fire_x = self.thermal.get_fire_data()
        if temp_max > FIRE_ARRIVE_TEMP:
            self.state_start_time, self.state = time.time(), 'CLOSE_IN'
            return
        if temp_max < FIRE_DETECT_TEMP - 5:
            print(f"   -> [APPROACH] 불씨 사라짐 ({temp_max:.1f}C) → PATROL 복귀")
            self.drive.stop()
            self.path, self.state = [], 'PATROL'
            return
        error = fire_x - 15
        self.drive.set_speed(0.20 + error * 0.01, 0.20 - error * 0.01)

    def _state_close_in(self):
        if time.time() - self.state_start_time < 1.5: self.drive.set_speed(0.30, 0.30)
        else:
            self.drive.stop()
            self.state_start_time, self.state = time.time(), 'DEPLOY'

    def _state_deploy(self):
        if time.time() - self.state_start_time < 3.3: self.actuator.move('extend', 1.0)
        else:
            self.actuator.stop()
            self.state_start_time, self.state = time.time(), 'EXTINGUISH'

    def _state_extinguish(self):
        if time.time() - self.state_start_time < 8.0:
            self.drive.set_speed(0.20, 0.20)
            self.arduino.start_pump()
            self.combine.run('suction', 0.60)
        else:
            self.drive.stop()
            self.arduino.stop_pump()
            self.combine.stop()
            self.state_start_time, self.state = time.time(), 'RESET'

    def _state_reset(self):
        if time.time() - self.state_start_time < 3.3:
            self.actuator.move('retract', 1.0)
            self.combine.run('discharge', 0.60)
        else:
            self.actuator.stop()
            self.combine.stop()
            self.path, self.state = [], 'PATROL'

    def _emergency_stop(self):
        print("\n[EMERGENCY STOP]")
        self.drive.stop()
        self.actuator.stop()
        self.combine.stop()
        self.arduino.stop_pump()
        self.rgb_cam.stop()
        cv2.destroyAllWindows()
        if hasattr(self, 'scan_iterator') and self.scan_iterator:
            try:
                self.lidar.stop()
                self.lidar.set_motor_pwm(0)
                self.lidar.disconnect()
            except Exception: pass

if __name__ == "__main__":
    FireFighterFSM().run()
