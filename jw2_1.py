import os
import json
import time
import threading
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DEFAULT_SAMPLE_RATE = 44100
BLOCK_SIZE          = 512
CHANNELS            = 2
ALPHA_SMOOTH        = 0.04      
TRAIL_LENGTH        = 28        
PRESET_DIR          = os.path.expanduser("~/.touchless_presets")


# ─────────────────────────────────────────────
# SMOOTH PARAMETER
# ─────────────────────────────────────────────
class SmoothParam:
    def __init__(self, init=0.0, alpha=ALPHA_SMOOTH):
        self.value  = float(init)
        self.alpha  = alpha
        self._target = float(init)

    def set_target(self, target):
        self._target = float(target)
        self.value  += self.alpha * (self._target - self.value)
        return self.value

    def tick(self):
        self.value += self.alpha * (self._target - self.value)
        return self.value

    def __float__(self):
        return self.value


# ─────────────────────────────────────────────
# DSP CLASSES
# ─────────────────────────────────────────────
class BiquadFilter:
    def __init__(self, samplerate=DEFAULT_SAMPLE_RATE, filter_type="lp"):
        self.filter_type = filter_type
        self.samplerate  = samplerate
        self._cutoff     = 1000.0
        self._q          = 0.707
        self.z1 = np.zeros((1, CHANNELS))
        self.z2 = np.zeros((1, CHANNELS))
        self._dirty = True
        self._calculate_coeffs()

    @property
    def cutoff(self): return self._cutoff
    @cutoff.setter
    def cutoff(self, v):
        v = float(np.clip(v, 20, 18000))
        if abs(v - self._cutoff) > 0.5:
            self._cutoff = v
            self._dirty  = True

    @property
    def q(self): return self._q
    @q.setter
    def q(self, v):
        v = float(np.clip(v, 0.1, 5.0))
        if abs(v - self._q) > 0.001:
            self._q     = v
            self._dirty = True

    def _calculate_coeffs(self):
        w0     = 2 * np.pi * self._cutoff / self.samplerate
        alpha  = np.sin(w0) / (2 * self._q)
        cos_w0 = np.cos(w0)
        if self.filter_type == "lp":
            b0 = (1 - cos_w0) / 2
            b1 =  1 - cos_w0
            b2 = (1 - cos_w0) / 2
        else:
            b0 =  (1 + cos_w0) / 2
            b1 = -(1 + cos_w0)
            b2 =  (1 + cos_w0) / 2
        a0 = 1 + alpha
        self.b0 = b0 / a0;  self.b1 = b1 / a0;  self.b2 = b2 / a0
        self.a1 = (-2 * cos_w0) / a0;            self.a2 = (1 - alpha) / a0
        self._dirty = False

    def process(self, data):
        if self._dirty:
            self._calculate_coeffs()
        out = np.zeros_like(data)
        for i in range(len(data)):
            x = data[i:i+1]
            y = self.b0 * x + self.z1
            self.z1 = self.b1 * x - self.a1 * y + self.z2
            self.z2 = self.b2 * x - self.a2 * y
            out[i] = y
        return out

class Distortion:
    def __init__(self, drive=1.0): self.drive = drive
    def process(self, data):
        d   = float(self.drive)
        sat = np.tanh(data * d)
        comp = 1.0 / (1.0 + np.log1p(d * 0.5))
        return sat * comp

class Delay:
    def __init__(self, samplerate=DEFAULT_SAMPLE_RATE, max_delay_sec=1.0):
        self.samplerate = samplerate
        self.buffer  = np.zeros((int(samplerate * max_delay_sec), CHANNELS))
        self.ptr     = 0
        self.current_delay_samples = int(samplerate * 0.3)

    def process(self, data, mix=0.3, feedback=0.4):
        out = np.zeros_like(data)
        for i in range(len(data)):
            read_idx   = (self.ptr - self.current_delay_samples) % len(self.buffer)
            delayed    = self.buffer[read_idx]
            self.buffer[self.ptr] = data[i] + delayed * feedback
            out[i]     = data[i] * (1.0 - mix) + delayed * mix
            self.ptr   = (self.ptr + 1) % len(self.buffer)
        return out

class SchroederReverb:
    COMB_L   = [1557, 1617, 1491, 1422]
    COMB_R   = [1592, 1668, 1525, 1456]  
    AP_DELAYS = [225, 341]

    def __init__(self):
        self.size = 0.5
        self._comb_l  = [np.zeros(d) for d in self.COMB_L]
        self._comb_r  = [np.zeros(d) for d in self.COMB_R]
        self._comb_pl = [0] * 4
        self._comb_pr = [0] * 4
        self._ap_l    = [np.zeros(d) for d in self.AP_DELAYS]
        self._ap_r    = [np.zeros(d) for d in self.AP_DELAYS]
        self._ap_pl   = [0] * 2
        self._ap_pr   = [0] * 2

    def _process_channel(self, mono, comb_bufs, comb_ptrs, ap_bufs, ap_ptrs, decay):
        n  = len(mono)
        out = np.zeros(n)
        for ci, (buf, ptr) in enumerate(zip(comb_bufs, comb_ptrs)):
            blen = len(buf)
            for i in range(n):
                idx = ptr % blen
                delayed = buf[idx]
                buf[idx] = mono[i] + delayed * decay
                out[i] += delayed
                ptr = (ptr + 1) % blen
            comb_ptrs[ci] = ptr
        out /= 4.0
        for ai, (buf, ptr) in enumerate(zip(ap_bufs, ap_ptrs)):
            blen = len(buf)
            for i in range(n):
                idx     = ptr % blen
                delayed = buf[idx]
                buf[idx] = out[i] + delayed * 0.5
                out[i]  = delayed - out[i] * 0.5
                ptr = (ptr + 1) % blen
            ap_ptrs[ai] = ptr
        return out

    def process(self, data, mix=0.3):
        if mix < 0.001: return data
        decay = 0.4 + self.size * 0.45   
        wet_l = self._process_channel(data[:, 0], self._comb_l, self._comb_pl, self._ap_l, self._ap_pl, decay)
        wet_r = self._process_channel(data[:, 1], self._comb_r, self._comb_pr, self._ap_r, self._ap_pr, decay)
        wet = np.stack([wet_l, wet_r], axis=1)
        return data * (1.0 - mix) + wet * mix

class SynthOscillator:
    def __init__(self, samplerate=DEFAULT_SAMPLE_RATE):
        self.samplerate = samplerate
        self.phase_l = 0.0
        self.phase_r = 0.025   

    def process(self, num_samples, freq):
        t        = np.arange(num_samples)
        phase_l  = self.phase_l + 2 * np.pi * freq * (t / self.samplerate)
        phase_r  = self.phase_r + 2 * np.pi * freq * 1.002 * (t / self.samplerate)  
        self.phase_l = phase_l[-1] % (2 * np.pi)
        self.phase_r = phase_r[-1] % (2 * np.pi)
        sig_l = (0.4 * np.sin(phase_l) + 0.3 * np.sin(2.01 * phase_l) + 0.2 * np.sin(3.02 * phase_l))
        sig_r = (0.4 * np.sin(phase_r) + 0.3 * np.sin(2.01 * phase_r) + 0.2 * np.sin(3.02 * phase_r))
        return np.stack([sig_l, sig_r], axis=1)

class Arpeggiator:
    def __init__(self, scale, samplerate=DEFAULT_SAMPLE_RATE):
        self.scale        = scale
        self.samplerate   = samplerate
        self.current_step = 0
        self.timer        = 0
        self.current_note = scale[0]

    def update(self, bpm):
        threshold = int(self.samplerate / (bpm / 60.0 * 8))
        self.timer += BLOCK_SIZE
        if self.timer >= threshold:
            self.timer        = 0
            self.current_step = (self.current_step + 1) % len(self.scale)
            self.current_note = self.scale[self.current_step]
        return self.current_note


# ─────────────────────────────────────────────
# GESTURE ENGINE & VISUALS
# ─────────────────────────────────────────────
class GestureEngine:
    TIP_IDS   = [4,  8, 12, 16, 20]
    JOINT_IDS = [3,  6, 10, 14, 18]

    def count_fingers(self, lms):
        up = []
        up.append(lms[4].x < lms[3].x)
        for tip, joint in zip(self.TIP_IDS[1:], self.JOINT_IDS[1:]):
            up.append(lms[tip].y < lms[joint].y)
        return up

    def classify(self, lms):
        up    = self.count_fingers(lms)
        n     = sum(up)
        pinch = np.hypot(lms[8].x - lms[4].x, lms[8].y - lms[4].y)
        if n == 0: return "FIST"
        if n == 5: return "OPEN"
        if n == 1 and up[1]: return "POINT"
        if n == 2 and up[1] and up[2]: return "PEACE"
        if pinch < 0.05: return "PINCH"
        return "FREE"


class HandTrail:
    def __init__(self, length=TRAIL_LENGTH, color=(0, 255, 65)):
        self.points = deque(maxlen=length)
        self.color  = color

    def add(self, pt): self.points.append(pt)
    def draw(self, canvas):
        n = len(self.points)
        if n < 2: return
        for i, pt in enumerate(self.points):
            alpha  = (i + 1) / n
            radius = max(1, int(alpha * 5))
            c      = tuple(int(ch * alpha) for ch in self.color)
            cv2.circle(canvas, pt, radius, c, -1)


class PresetManager:
    def __init__(self, directory=PRESET_DIR):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.slots: dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        for f in os.listdir(self.directory):
            if f.endswith(".json"):
                name = f[:-5]
                try:
                    with open(os.path.join(self.directory, f)) as fh:
                        self.slots[name] = json.load(fh)
                except: pass

    def save(self, name, params: dict):
        self.slots[name] = dict(params)
        with open(os.path.join(self.directory, f"{name}.json"), "w") as fh:
            json.dump(self.slots[name], fh, indent=2)

    def load(self, name) -> dict: return dict(self.slots.get(name, {}))
    def list_presets(self) -> list[str]: return sorted(self.slots.keys())


# ─────────────────────────────────────────────
# MAIN CONTROLLER
# ─────────────────────────────────────────────
class TouchlessWorkstation:
    SCALES = {
        "PENTA": [261.63, 293.66, 349.23, 392.00, 440.00, 523.25],
        "MINOR": [261.63, 293.66, 311.13, 349.23, 392.00, 415.30, 466.16, 523.25],
        "BLUES": [261.63, 311.13, 349.23, 369.99, 392.00, 466.16, 523.25],
        "MAJOR": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25],
    }

    _PARAM_DEFAULTS = {
        "Master Vol":    0.7,
        "Filter Cutoff": 0.5,
        "Filter Res":    0.2,
        "Distortion":    0.0,
        "Glitch":        0.0,   
        "Delay Mix":     0.0,
        "Reverb Mix":    0.0,
        "Synth Vol":     0.5,
        "Arp BPM":       0.4,
    }

    def __init__(self):
        self.mode          = "EFFECTOR"
        self.scale_name    = "PENTA"
        self.running       = False
        self.params_locked = False
        self.precision_mode = False
        self.samplerate    = DEFAULT_SAMPLE_RATE  # 기본 설정 후 오디오 디바이스 연동 시 동적 업데이트됨

        self._lock    = threading.Lock()
        self._smooth  = {k: SmoothParam(v) for k, v in self._PARAM_DEFAULTS.items()}
        self._targets = dict(self._PARAM_DEFAULTS)

        # DSP 컴포넌트 초기화 (동적 샘플레이트 적용 예정)
        self.filter  = BiquadFilter(self.samplerate)
        self.dist    = Distortion()
        self.delay   = Delay(self.samplerate)
        self.reverb  = SchroederReverb()
        self.synth   = SynthOscillator(self.samplerate)
        self.arp     = Arpeggiator(self.SCALES[self.scale_name], self.samplerate)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75, min_tracking_confidence=0.6)
        self.gesture_engine = GestureEngine()
        self.trails = [HandTrail(color=(0, 255, 65)), HandTrail(color=(200, 50, 255))]

        self._hold_start: dict[str, float] = {}
        self.presets = PresetManager()
        self.ascii_chars = " .:-=+*#%@"
        self.ascii_scale = 10
        
        # 버튼 영역 좌표 저장용 (마우스 인터랙션 바인딩용)
        self.buttons_hitboxes = {}

    def _set_target(self, key: str, val: float):
        if self.params_locked: return
        with self._lock:
            mult = 0.2 if self.precision_mode else 1.0
            cur  = self._targets[key]
            self._targets[key] = float(np.clip(cur + (float(val) - cur) * mult, 0.0, 1.0))

    def _tick_smooth(self):
        with self._lock:
            for k, sp in self._smooth.items(): sp.set_target(self._targets[k])

    def _get_smooth(self) -> dict:
        with self._lock: return {k: sp.value for k, sp in self._smooth.items()}

    def _reset_params(self):
        with self._lock: self._targets = dict(self._PARAM_DEFAULTS)

    @staticmethod
    def format_param(name: str, val: float) -> str:
        if   name == "Filter Cutoff": return f"{20 * (1000 ** val):.0f} Hz"
        elif name == "Filter Res":    return f"Q {0.1 + val * 4.9:.2f}"
        elif name == "Distortion":    return f"{val * 28:.1f} dB"
        elif name == "Glitch":        return f"{val * 100:.0f}%"
        elif name == "Delay Mix":     return f"{val * 100:.0f}%"
        elif name == "Reverb Mix":    return f"{val * 100:.0f}%"
        elif name == "Synth Vol":     return f"{val * 100:.0f}%"
        elif name == "Arp BPM":       return f"{60 + val * 180:.0f} BPM"
        elif name == "Master Vol":    return f"{val * 100:.0f}%"
        return f"{val:.2f}"

    def _reactive_color(self) -> tuple:
        p = self._get_smooth()
        r = int(p["Distortion"]  * 220)
        g = int(p["Filter Cutoff"] * 180 + 60)
        b = int(p["Reverb Mix"]  * 180)
        if self.mode == "SYNTH": return (min(b + 120, 255), 20, min(r + 100, 255))
        return (0, g, r)                                         

    def audio_callback(self, indata, outdata, frames, time_info, status):
        self._tick_smooth()
        p = self._get_smooth()

        if self.mode == "EFFECTOR" and indata is not None and indata.shape[1] >= CHANNELS:
            sig = indata.copy()
        else:
            bpm  = 60 + p["Arp BPM"] * 180
            note = self.arp.update(bpm)
            sig  = self.synth.process(frames, note) * p["Synth Vol"]

        self.dist.drive = 1.0 + p["Distortion"] * 25.0
        sig = self.dist.process(sig)

        self.filter.cutoff = 20 * (1000 ** p["Filter Cutoff"])
        self.filter.q      = 0.1 + p["Filter Res"] * 4.9
        sig = self.filter.process(sig)
        sig *= 1.0 / (1.0 + p["Filter Res"] * 0.7)

        sig = self.delay.process(sig, mix=p["Delay Mix"], feedback=p["Delay Mix"] * 0.7)
        self.reverb.size = p["Reverb Mix"]
        sig = self.reverb.process(sig, mix=p["Reverb Mix"])

        outdata[:] = np.clip(sig * p["Master Vol"], -1.0, 1.0)

    def render_ascii(self, frame: np.ndarray) -> np.ndarray:
        h, w, _ = frame.shape
        capture_w = w - 240
        cropped = frame[:, :capture_w]
        
        small = cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), (capture_w // self.ascii_scale, h // self.ascii_scale))
        small  = cv2.equalizeHist(small)
        canvas = np.zeros_like(frame)
        color  = self._reactive_color()
        cw, ch = self.ascii_scale, self.ascii_scale
        n      = len(self.ascii_chars)
        glitch = self._smooth["Glitch"].value
        rows, cols = small.shape

        for i in range(rows):
            shift = 0
            if glitch > 0.05 and np.random.random() < glitch * 0.6:
                shift = int(np.random.uniform(-glitch * 8, glitch * 8))
            row_color = color
            if glitch > 0.2 and np.random.random() < glitch * 0.25:
                row_color = tuple(255 - c for c in color)

            for j in range(cols):
                src_j = min(max(j + shift, 0), cols - 1)
                idx   = min(int(small[i, src_j] / 256 * n), n - 1)
                if glitch > 0.4 and np.random.random() < glitch * 0.15: idx = np.random.randint(0, n)
                cv2.putText(canvas, self.ascii_chars[idx], (j * cw, i * ch + ch), cv2.FONT_HERSHEY_PLAIN, 0.7, row_color, 1)

        if glitch > 0.5:
            for _ in range(int(glitch * 6)):
                by = np.random.randint(0, h)
                bh = np.random.randint(2, 8)
                canvas[by:by+bh, :capture_w] = np.clip(canvas[by:by+bh, :capture_w].astype(int) + 60, 0, 255).astype(np.uint8)
        return canvas

    def _check_hold(self, gesture: str, threshold: float) -> bool:
        now = time.time()
        if gesture not in self._hold_start:
            self._hold_start[gesture] = now
            return False
        return (now - self._hold_start[gesture]) >= threshold

    def _clear_hold(self, gesture: str): self._hold_start.pop(gesture, None)

    def _get_control_zones(self, capture_w: int, h: int) -> tuple[tuple, tuple]:
        """왼쪽/오른쪽 화면 중앙의 손 제어 구역 (x1, y1, x2, y2) 반환."""
        ui_top    = 55
        ui_bottom = 55
        usable_h  = h - ui_top - ui_bottom
        half_w    = capture_w // 2
        zone_w    = int(half_w * 0.55)
        zone_h    = int(usable_h * 0.55)
        cy        = ui_top + usable_h // 2

        left_cx  = half_w // 2
        right_cx = half_w + half_w // 2

        left_zone  = (left_cx - zone_w // 2, cy - zone_h // 2, left_cx + zone_w // 2, cy + zone_h // 2)
        right_zone = (right_cx - zone_w // 2, cy - zone_h // 2, right_cx + zone_w // 2, cy + zone_h // 2)
        return left_zone, right_zone

    @staticmethod
    def _point_in_zone(x: int, y: int, zone: tuple) -> bool:
        x1, y1, x2, y2 = zone
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _norm_in_zone(x: int, y: int, zone: tuple) -> tuple[float, float]:
        x1, y1, x2, y2 = zone
        norm_x = float(np.clip((x - x1) / max(x2 - x1, 1), 0.0, 1.0))
        norm_y = float(np.clip(1.0 - (y - y1) / max(y2 - y1, 1), 0.0, 1.0))
        return norm_x, norm_y

    def _draw_control_zone(self, canvas, zone: tuple, label: str, base_color: tuple, active: bool):
        x1, y1, x2, y2 = zone
        color = base_color if active else tuple(max(c // 3, 20) for c in base_color)
        thickness = 3 if active else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.08 if active else 0.03, canvas, 0.92 if active else 0.97, 0, canvas)
        cv2.putText(canvas, label, (x1 + 8, y1 + 18), cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1, cv2.LINE_AA)

    def on_mouse_click(self, event, x, y, flags, param):
        """[NEW] 모바일 UI 메뉴 영역 마우스 클릭 이벤트 처리 루틴"""
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn_id, (bx1, by1, bx2, by2) in self.buttons_hitboxes.items():
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    if btn_id == "EXIT":
                        print("[INFO] 마우스 버튼 클릭 종료 트리거.")
                        self.running = False
                    else:
                        self.mode = btn_id
                        print(f"[INFO] 마우스 메뉴 클릭 전환 → {self.mode}")

    def process_gestures(self, results, frame_shape, canvas):
        h, w  = frame_shape[:2]
        ge    = self.gesture_engine
        capture_w = w - 240
        left_zone, right_zone = self._get_control_zones(capture_w, h)
        left_active = right_active = False

        if not results.multi_hand_landmarks:
            self._draw_control_zone(canvas, left_zone,  "LEFT  | DELAY / REVERB / SYNTH", (255, 180, 50),  False)
            self._draw_control_zone(canvas, right_zone, "RIGHT | FILTER / DISTORTION",    (50, 255, 120), False)
            self._hold_start.clear()
            self.precision_mode = False
            return

        lms_list = results.multi_hand_landmarks
        handedness_list = results.multi_handedness or []
        gestures = [ge.classify(lm.landmark) for lm in lms_list]
        self.precision_mode = any(g == "POINT" for g in gestures)

        if any(g == "OPEN" for g in gestures):
            if self._check_hold("OPEN", 1.0):
                self._reset_params()
                self._clear_hold("OPEN")
        else: self._clear_hold("OPEN")

        if any(g == "PEACE" for g in gestures):
            if self._check_hold("PEACE", 0.5):
                self.mode = "SYNTH" if self.mode == "EFFECTOR" else "EFFECTOR"
                self._clear_hold("PEACE")
        else: self._clear_hold("PEACE")

        for idx, (hand_lms, gesture) in enumerate(zip(lms_list, gestures)):
            lms = hand_lms.landmark
            hand_label = "Right"
            if idx < len(handedness_list):
                hand_label = handedness_list[idx].classification[0].label

            is_left_hand  = hand_label == "Left"
            zone          = left_zone if is_left_hand else right_zone
            trail         = self.trails[0 if is_left_hand else 1]
            zone_color    = (255, 180, 50) if is_left_hand else (50, 255, 120)

            itip, ttip = lms[8], lms[4]
            ix, iy = int(itip.x * w), int(itip.y * h)
            tx, ty = int(ttip.x * w), int(ttip.y * h)

            if ix > capture_w: continue

            in_zone = self._point_in_zone(ix, iy, zone)
            if is_left_hand:
                left_active  = left_active or in_zone
            else:
                right_active = right_active or in_zone

            trail.add((ix, iy))
            trail.draw(canvas)

            for pt_idx in [0, 4, 8, 12, 16, 20]:
                pt = lms[pt_idx]
                marker_color = zone_color if in_zone else (100, 100, 100)
                cv2.drawMarker(canvas, (int(pt.x * w), int(pt.y * h)), marker_color, cv2.MARKER_CROSS, 10, 1)

            lbl_y = max(int(lms[0].y * h) - 22, 18)
            status = gesture if in_zone else f"{gesture} (OUT)"
            cv2.putText(canvas, status, (int(lms[0].x * w) - 24, lbl_y), cv2.FONT_HERSHEY_PLAIN, 1.1, zone_color if in_zone else (120, 120, 120), 1)

            if not in_zone:
                continue

            norm_x, iy_norm = self._norm_in_zone(ix, iy, zone)

            if is_left_hand:
                if gesture == "PINCH":
                    self._set_target("Delay Mix", norm_x)
                    self._set_target("Reverb Mix", iy_norm)
                    cv2.line(canvas, (ix, iy), (tx, ty), (255, 255, 255), 1)
                    cv2.circle(canvas, (ix, iy), 18, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(canvas, (ix, iy), 6, zone_color, -1)
                elif gesture == "FIST":
                    self._set_target("Arp BPM", norm_x)
                    self._set_target("Synth Vol", iy_norm)
            else:
                if gesture == "PINCH":
                    self._set_target("Filter Res", norm_x)
                    self._set_target("Filter Cutoff", iy_norm)
                    cv2.line(canvas, (ix, iy), (tx, ty), (255, 255, 255), 1)
                    cv2.circle(canvas, (ix, iy), 18, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.circle(canvas, (ix, iy), 6, zone_color, -1)
                elif gesture == "FIST":
                    self._set_target("Distortion", iy_norm)
                    self._set_target("Glitch", norm_x)

            if gesture == "PINCH":
                sx, sy, sw = capture_w // 2 - 100, h - 50, 200
                if sx <= ix <= sx + sw and sy - 20 <= iy <= sy + 20:
                    with self._lock: self._targets["Master Vol"] = float(np.clip((ix - sx) / sw, 0.0, 1.0))

        self._draw_control_zone(canvas, left_zone,  "LEFT  | DELAY / REVERB / SYNTH", (255, 180, 50),  left_active)
        self._draw_control_zone(canvas, right_zone, "RIGHT | FILTER / DISTORTION",    (50, 255, 120), right_active)

    def draw_ui(self, frame, w, h):
        color = self._reactive_color()
        p     = self._get_smooth()
        menu_x = w - 240

        cv2.line(frame, (menu_x, 0), (menu_x, h), color, 1)

        # ── 1. 상단 블루/네온 스타일 게임 타이틀 배너 ──
        cv2.rectangle(frame, (0, 0), (w, 55), (130, 50, 20), -1) 
        cv2.line(frame, (0, 55), (w, 55), (255, 255, 255), 1)
        cv2.putText(frame, "TOUCHLESS WORKSTATION", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # ── 2. 우측 모바일 3단 메뉴 버튼 레이아웃 ──
        btn_w, btn_h = 200, 50
        btn_start_y = 90
        gap = 25

        menu_buttons = [
            {"id": "EFFECTOR", "label": "EFFECT MODE"},
            {"id": "SYNTH",    "label": "SYNTH MODE"},
            {"id": "EXIT",     "label": "GAME EXIT"}
        ]

        for i, btn in enumerate(menu_buttons):
            bx = menu_x + 20
            by = btn_start_y + i * (btn_h + gap)
            
            # 마우스 클릭 판정용 힛박스 딕셔너리 적재
            self.buttons_hitboxes[btn["id"]] = (bx, by, bx + btn_w, by + btn_h)
            
            is_active = (self.mode == btn["id"])
            bg_color = tuple(int(c * 0.9) for c in color) if is_active else (25, 25, 25)
            text_color = (255, 255, 255) if is_active else tuple(int(c * 0.5) for c in color)
            
            cv2.rectangle(frame, (bx, by), (bx + btn_w, by + btn_h), bg_color, -1)
            cv2.rectangle(frame, (bx, by), (bx + btn_w, by + btn_h), color, 1)
            cv2.putText(frame, btn["label"], (bx + 25, by + 32), cv2.FONT_HERSHEY_DUPLEX, 0.6, text_color, 1, cv2.LINE_AA)

        # ── 3. 좌측 오디오 제어 파라미터 상태창 피드백 ──
        y_off = 95
        for name, sp in self._smooth.items():
            if name == "Master Vol": continue
            val = sp.value
            bar_w = int(val * 110)
            label = f"{name.upper()}: {self.format_param(name, val)}"
            
            cv2.putText(frame, label, (20, y_off), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1)
            cv2.rectangle(frame, (170, y_off - 10), (170 + 110, y_off + 2), (30, 30, 30), -1)
            cv2.rectangle(frame, (170, y_off - 10), (170 + bar_w, y_off + 2), color, -1)
            y_off += 26

        # ── 4. 하단 마스터 볼륨 바 ──
        mv = p["Master Vol"]
        sx, sy = (menu_x // 2) - 100, h - 45
        cv2.rectangle(frame, (sx, sy), (sx + 200, sy + 12), (30, 30, 30), -1)
        cv2.rectangle(frame, (sx, sy), (sx + int(mv * 200), sy + 12), color, -1)
        cv2.putText(frame, f"MASTER VOL: {int(mv*100)}%", (sx, sy - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1)

        # ── 5. 우측 하단 프리셋 정보 모듈 ──
        cv2.putText(frame, f"SCALE: {self.scale_name}", (menu_x + 25, h - 110), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1)
        plist = self.presets.list_presets()
        if plist:
            cv2.putText(frame, "SLOTS (1-5):", (menu_x + 25, h - 85), cv2.FONT_HERSHEY_PLAIN, 0.75, color, 1)
            for pi, pname in enumerate(plist[:3]):
                cv2.putText(frame, f" P{pi+1}: {pname[-10:]}", (menu_x + 25, h - 65 + pi * 15), cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1)

    def _find_audio_devices(self) -> tuple[int, int, int]:
        """[NEW] 오디오 장치 정보 및 디바이스 네이티브 샘플레이트 동적 조회"""
        devices  = sd.query_devices()
        in_idx, out_idx  = None, None
        try:
            default_in, default_out = sd.default.device[0], sd.default.device[1]
        except: default_in, default_out = 0, 0

        OUT_NAMES = ["스피커", "speakers", "built-in output", "headphones", "headset", "default", "realtek"]
        for i, d in enumerate(devices):
            name_l = d['name'].lower()
            if ("blackhole" in name_l or "virtual" in name_l or "cable" in name_l) and d['max_input_channels'] > 0:
                if in_idx is None: in_idx = i
            if d['max_output_channels'] > 0:
                for n in OUT_NAMES:
                    if n in name_l and out_idx is None: out_idx = i

        chosen_in = in_idx if in_idx is not None else default_in
        chosen_out = out_idx if out_idx is not None else default_out
        
        # [NEW] 하드웨어의 타겟 샘플레이트 추적 및 보정
        s_rate = DEFAULT_SAMPLE_RATE
        try:
            dev_info = sd.query_devices(chosen_out)
            s_rate = int(dev_info.get('default_samplerate', DEFAULT_SAMPLE_RATE))
        except: pass
        
        return chosen_in, chosen_out, s_rate

    def run(self):
        in_idx, out_idx, self.samplerate = self._find_audio_devices()
        
        # [NEW] 동적 매칭된 샘플 레이트를 각 오디오 DSP 클래스 계수에 동기화 적용
        self.filter.samplerate = self.samplerate
        self.delay.samplerate = self.samplerate
        self.synth.samplerate = self.samplerate
        self.arp.samplerate = self.samplerate

        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return

        # 마우스 콜백 바인딩을 위한 명식적 윈도우 이름 지정 선언
        win_name = "TOUCHLESS_WORKSTATION_v12"
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, self.on_mouse_click)

        self.running = True
        stream_kwargs = {"device": (in_idx, out_idx), "samplerate": self.samplerate, "blocksize": BLOCK_SIZE, "channels": CHANNELS, "callback": self.audio_callback}

        try:
            stream = sd.Stream(**stream_kwargs)
            stream.start()
            print(f"[AUDIO_OK] 양방향 오디오 가동 완료 (Sample Rate: {self.samplerate}Hz)")
        except:
            try:
                self.mode = "SYNTH"
                stream = sd.OutputStream(device=out_idx, samplerate=self.samplerate, blocksize=BLOCK_SIZE, channels=CHANNELS, callback=self.audio_callback)
                stream.start()
                print(f"[AUDIO_WARN] 출력 전용 모드 전환 오픈 완료 ({self.samplerate}Hz)")
            except:
                cap.release()
                return

        try:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret: continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                canvas = self.render_ascii(frame)
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                self.process_gestures(results, frame_shape=frame.shape, canvas=canvas)

                self.draw_ui(canvas, w, h)
                cv2.imshow(win_name, canvas)
                key = cv2.waitKey(1) & 0xFF

                if key == 27: self.running = False
                elif key == ord('m'): self.mode = "SYNTH" if self.mode == "EFFECTOR" else "EFFECTOR"
                elif key == ord('n'):
                    keys = list(self.SCALES.keys())
                    self.scale_name = keys[(keys.index(self.scale_name) + 1) % len(keys)]
                    self.arp.scale  = self.SCALES[self.scale_name]
                elif key == ord('s'): self.presets.save(f"preset_{int(time.time())}", self._get_smooth())
                elif ord('1') <= key <= ord('5'):
                    slot = key - ord('1')
                    plist = self.presets.list_presets()
                    if slot < len(plist):
                        loaded = self.presets.load(plist[slot])
                        with self._lock:
                            for k, v in loaded.items():
                                if k in self._targets: self._targets[k] = float(v)

        except KeyboardInterrupt: pass
        finally:
            self.running = False
            try:
                stream.stop()
                stream.close()
            except: pass
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    TouchlessWorkstation().run()