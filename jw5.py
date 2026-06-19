"""
TOUCHLESS WORKSTATION v12.MOUSE_HYBRID
─────────────────────────────────────────────────────────────
반영 및 수정 사항 요약:
  1. 맥북 사운드디바이스 문법 수정 반영 — channels=(1, 2) 튜플 형태 스트림 오픈 적용 (PortAudio -9998 에러 예방)
  2. 오리지널 터치리스 기능 완전 복원 — ASCII 그래픽 렌더러, 0.5초 제스처 타이머, 두 손 거리 마스터 볼륨 등 포함
  3. 마우스 노브 컨트롤 서브 시스템 — 우측 패널의 각 파라미터 게이지 바를 클릭/드래그하여 미세 조정 가능
  4. 단축키 시스템 유지 — S(프리셋 저장), N(스케일 전환), 1~5(프리셋 로드), Backspace(메뉴)
"""

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
ALPHA_SMOOTH        = 0.05      
TRAIL_LENGTH        = 20        
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
# DSP COMPONENTS
# ─────────────────────────────────────────────
class BiquadFilter:
    def __init__(self, samplerate=DEFAULT_SAMPLE_RATE, filter_type="lp"):
        self.filter_type = filter_type
        self.samplerate  = samplerate
        self._cutoff     = 1000.0
        self._q          = 0.707
        self.z1 = np.zeros((1, 2))  # Stereo output buffer
        self.z2 = np.zeros((1, 2))
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
        self.buffer  = np.zeros((int(samplerate * max_delay_sec), 2))
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
# GESTURE ENGINE
# ─────────────────────────────────────────────
class GestureEngine:
    TIP_IDS   = [4,  8, 12, 16, 20]
    JOINT_IDS = [3,  6, 10, 14, 18]

    def classify(self, lms):
        up = []
        up.append(lms[4].x < lms[3].x)
        for tip, joint in zip(self.TIP_IDS[1:], self.JOINT_IDS[1:]):
            up.append(lms[tip].y < lms[joint].y)
            
        n = sum(up)
        pinch_dist = np.hypot(lms[8].x - lms[4].x, lms[8].y - lms[4].y)
        
        if pinch_dist < 0.05: return "PINCH"
        if n == 0: return "FIST"
        if n == 5: return "OPEN"
        if n == 1 and up[1]: return "POINT"
        if n == 2 and up[1] and up[2]: return "PEACE"
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
            radius = max(1, int(alpha * 4))
            c = tuple(int(ch * alpha) for ch in self.color)
            cv2.circle(canvas, pt, radius, c, -1)


class PresetManager:
    def __init__(self, directory=PRESET_DIR):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.slots = {}
        self._load_all()

    def _load_all(self):
        for f in os.listdir(self.directory):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(self.directory, f)) as fh:
                        self.slots[f[:-5]] = json.load(fh)
                except: pass

    def save(self, name, params: dict):
        self.slots[name] = dict(params)
        with open(os.path.join(self.directory, f"{name}.json"), "w") as fh:
            json.dump(self.slots[name], fh, indent=2)

    def load(self, name) -> dict: return dict(self.slots.get(name, {}))
    def list_presets(self) -> list[str]: return sorted(self.slots.keys())


# ─────────────────────────────────────────────
# MAIN WORKSTATION CONTROLLER
# ─────────────────────────────────────────────
class TouchlessWorkstation:
    SCALES = {
        "PENTA": [261.63, 293.66, 349.23, 392.00, 440.00, 523.25],
        "MINOR": [261.63, 293.66, 311.13, 349.23, 392.00, 415.30, 466.16, 523.25],
        "MAJOR": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25],
    }

    # 정렬 순서 고정을 위해 리스트로 정의
    PARAM_NAMES = [
        "Filter Cutoff", "Filter Res", "Distortion", "Glitch", 
        "Delay Mix", "Reverb Mix", "Synth Vol", "Arp BPM", "Master Vol"
    ]

    _PARAM_DEFAULTS = {
        "Filter Cutoff": 0.5,
        "Filter Res":    0.2,
        "Distortion":    0.0,
        "Glitch":        0.0,   
        "Delay Mix":     0.0,
        "Reverb Mix":    0.0,
        "Synth Vol":     0.5,
        "Arp BPM":       0.4,
        "Master Vol":    0.7,
    }

    def __init__(self):
        self.state          = "MAIN_MENU"  
        self.mode           = "EFFECTOR"
        self.scale_name     = "PENTA"
        self.running        = False
        self.precision_mode = False
        self.samplerate     = DEFAULT_SAMPLE_RATE

        self._lock    = threading.Lock()
        self._smooth  = {k: SmoothParam(v) for k, v in self._PARAM_DEFAULTS.items()}
        self._targets = dict(self._PARAM_DEFAULTS)

        self.filter  = BiquadFilter(self.samplerate)
        self.dist    = Distortion()
        self.delay   = Delay(self.samplerate)
        self.reverb  = SchroederReverb()
        self.synth   = SynthOscillator(self.samplerate)
        self.arp     = Arpeggiator(self.SCALES[self.scale_name], self.samplerate)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.gesture_engine = GestureEngine()
        self.trails = [HandTrail(color=(0, 255, 65)), HandTrail(color=(230, 50, 255))]
        
        self.buttons_hitboxes = {}
        self.knob_hitboxes = {}  # 마우스 노브 제어용 힛박스
        self.is_dragging_knob = None  # 현재 마우스로 드래그 중인 파라미터 이름

        self._hold_start = {}
        self._last_gestures = {}
        self.ascii_chars = " .:-=+*#%@"
        self.ascii_scale = 12

    def _set_target(self, key: str, val: float):
        with self._lock:
            cur = self._targets[key]
            mult = 0.2 if self.precision_mode else 1.0
            self._targets[key] = float(np.clip(cur + (float(val) - cur) * mult, 0.0, 1.0))

    def _tick_smooth(self):
        with self._lock:
            for k, sp in self._smooth.items(): sp.set_target(self._targets[k])

    def _get_smooth(self) -> dict:
        with self._lock: return {k: sp.value for k, sp in self._smooth.items()}

    @staticmethod
    def format_param(name: str, val: float) -> str:
        if   name == "Filter Cutoff": return f"{20 * (1000 ** val):.0f} Hz"
        elif name == "Filter Res":    return f"Q {0.1 + val * 4.9:.2f}"
        elif name == "Distortion":    return f"{val * 28:.1f} dB"
        elif name == "Arp BPM":       return f"{60 + val * 180:.0f} BPM"
        return f"{val * 100:.0f}%"

    def _reactive_color(self) -> tuple:
        p = self._get_smooth()
        if self.mode == "SYNTH": return (235, 30, 150)
        return (0, int(p["Filter Cutoff"] * 170 + 85), int(p["Distortion"] * 220))                                         

    def audio_callback(self, indata, outdata, frames, time_info, status):
        self._tick_smooth()
        p = self._get_smooth()

        if self.state != "WORKSTATION":
            outdata.fill(0)
            return

        if self.mode == "EFFECTOR" and indata is not None:
            sig = np.repeat(indata, 2, axis=1)  
        else:
            note = self.arp.update(60 + p["Arp BPM"] * 180)
            sig  = self.synth.process(frames, note) * p["Synth Vol"]

        sig = self.dist.process(sig)
        self.filter.cutoff = 20 * (1000 ** p["Filter Cutoff"])
        self.filter.q      = 0.1 + p["Filter Res"] * 4.9
        sig = self.filter.process(sig)

        sig = self.delay.process(sig, mix=p["Delay Mix"], feedback=p["Delay Mix"] * 0.6)
        self.reverb.size = p["Reverb Mix"]
        sig = self.reverb.process(sig, mix=p["Reverb Mix"])

        outdata[:] = np.clip(sig * p["Master Vol"], -1.0, 1.0)

    # ─────────────────────────────────────────────
    # MOUSE CONTROLLER (사이드바 및 하단 노브 인터랙션)
    # ─────────────────────────────────────────────
    def handle_mouse_knobs(self, x, y):
        if self.is_dragging_knob and self.is_dragging_knob in self.knob_hitboxes:
            x1, _, x2, _ = self.knob_hitboxes[self.is_dragging_knob]
            val = float(np.clip((x - x1) / (x2 - x1), 0.0, 1.0))
            self._set_target(self.is_dragging_knob, val)

    def on_mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. 사이드바 메뉴 버튼 클릭 핸들링
            for btn_id, (bx1, by1, bx2, by2) in self.buttons_hitboxes.items():
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    if btn_id == "MENU_START": self.state = "WORKSTATION"
                    elif btn_id == "MENU_EXIT" or btn_id == "GAME_EXIT": self.running = False
                    elif btn_id == "TOGGLE_EFF": self.mode = "EFFECTOR"
                    elif btn_id == "TOGGLE_SYN": self.mode = "SYNTH"
                    return

            # 2. 파라미터 게이지 바 클릭(드래그 시작) 핸들링
            for p_name, (kx1, ky1, kx2, ky2) in self.knob_hitboxes.items():
                if kx1 <= x <= kx2 and ky1 <= y <= ky2:
                    self.is_dragging_knob = p_name
                    val = float(np.clip((x - kx1) / (kx2 - kx1), 0.0, 1.0))
                    self._set_target(p_name, val)
                    return

        elif event == cv2.EVENT_MOUSEMOVE:
            # 드래그 중 이동 시 피드백 업데이트
            if self.is_dragging_knob:
                self.handle_mouse_knobs(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging_knob = None

    def render_main_menu(self, canvas, w, h):
        accent_color = (255, 140, 20)
        cv2.rectangle(canvas, (30, 30), (w - 30, h - 30), accent_color, 2)

        cv2.putText(canvas, "TOUCHLESS WORKSTATION", (w // 2 - 240, h // 2 - 110), cv2.FONT_HERSHEY_DUPLEX, 1.2, accent_color, 2, cv2.LINE_AA)
        cv2.putText(canvas, "SYSTEM CORE V12.MOUSE_HYBRID", (w // 2 - 190, h // 2 - 75), cv2.FONT_HERSHEY_PLAIN, 1.0, (150, 150, 150), 1, cv2.LINE_AA)

        btn_w, btn_h = 320, 50
        menu_configs = [
            {"id": "MENU_START", "label": "START WORKSTATION [1]"},
            {"id": "MENU_EXIT",  "label": "QUIT SYSTEM       [3]"}
        ]

        for i, config in enumerate(menu_configs):
            bx = w // 2 - btn_w // 2
            by = h // 2 + (i * 70)
            self.buttons_hitboxes[config["id"]] = (bx, by, bx + btn_w, by + btn_h)
            
            cv2.rectangle(canvas, (bx, by), (bx + btn_w, by + btn_h), (25, 25, 25), -1)
            cv2.rectangle(canvas, (bx, by), (bx + btn_w, by + btn_h), accent_color, 1)
            cv2.putText(canvas, config["label"], (bx + 35, by + 32), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, "Tip: You can use mouse clicks or num keys 1, 3", (w // 2 - 180, h - 60), cv2.FONT_HERSHEY_PLAIN, 0.9, (120, 120, 120), 1)

    def generate_ascii_art(self, frame, canvas, w, h, color):
        small = cv2.resize(frame, (w // self.ascii_scale, h // self.ascii_scale))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        for y in range(gray.shape[0]):
            for x in range(gray.shape[1]):
                val = gray[y, x]
                idx = int(val / 256 * len(self.ascii_chars))
                char = self.ascii_chars[min(idx, len(self.ascii_chars)-1)]
                px = x * self.ascii_scale
                py = y * self.ascii_scale
                if px < w - 240: 
                    cv2.putText(canvas, char, (px, py + self.ascii_scale), cv2.FONT_HERSHEY_PLAIN, 0.6, tuple(int(c * (val/255.0)) for c in color), 1)

    def render_workstation(self, frame, canvas, results, w, h):
        color = self._reactive_color()
        capture_w = w - 240

        # 아스키 비디오 아트 스트리밍 백그라운드 구현
        self.generate_ascii_art(frame, canvas, w, h, color)

        cv2.line(canvas, (capture_w // 2, 60), (capture_w // 2, h), (60, 60, 60), 1, cv2.LINE_AA)

        now = time.time()
        active_hands_positions = []

        # ── TOUCHLESS: VISION CONTROL ENGINE ──
        if results.multi_hand_landmarks:
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                lms = hand_lms.landmark
                raw_gesture = self.gesture_engine.classify(lms)
                
                if idx not in self._last_gestures or self._last_gestures[idx] != raw_gesture:
                    self._last_gestures[idx] = raw_gesture
                    self._hold_start[idx] = now
                
                gesture = raw_gesture if (now - self._hold_start.get(idx, now)) >= 0.5 else "HOLDING..."

                ix, iy = int(lms[8].x * w), int(lms[8].y * h)
                if ix > capture_w: continue
                active_hands_positions.append((ix, iy))

                trail = self.trails[idx % 2]
                trail.add((ix, iy))
                trail.draw(canvas)

                for joint in [4, 8, 12, 16, 20]:
                    cv2.circle(canvas, (int(lms[joint].x * w), int(lms[joint].y * h)), 4, color, -1)

                cv2.putText(canvas, gesture, (int(lms[0].x * w) - 20, int(lms[0].y * h) - 25), cv2.FONT_HERSHEY_PLAIN, 1.1, color, 1)

                if gesture != "HOLDING...":
                    norm_x = float(np.clip(lms[8].x * (w / capture_w), 0.0, 1.0))
                    norm_y = float(np.clip(1.0 - lms[8].y, 0.0, 1.0))

                    if norm_x > 0.5:  
                        if gesture == "PINCH":
                            self._set_target("Filter Res", (norm_x - 0.5) * 2)
                            self._set_target("Filter Cutoff", norm_y)
                        elif gesture == "FIST":
                            self._set_target("Distortion", norm_y)
                    else:  
                        if gesture == "PINCH":
                            self._set_target("Delay Mix", norm_x * 2)
                            self._set_target("Reverb Mix", norm_y)
                        elif gesture == "FIST":
                            self._set_target("Arp BPM", norm_x * 2)
                            self._set_target("Synth Vol", norm_y)

            # 두 손 거리 비례 마스터 볼륨 자동 연산
            if len(active_hands_positions) >= 2:
                pt1, pt2 = active_hands_positions[0], active_hands_positions[1]
                dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
                norm_dist = float(np.clip(dist / (capture_w * 0.5), 0.0, 1.0))
                self._set_target("Master Vol", norm_dist)

        # ── UI OVERLAY & MOUSE HITTING MAP ──
        cv2.rectangle(canvas, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.line(canvas, (0, 55), (w, 55), color, 1)
        cv2.putText(canvas, f"TOUCHLESS WORKSTATION [SCALE: {self.scale_name}]", (25, 36), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[Backspace: Menu]", (capture_w - 180, 34), cv2.FONT_HERSHEY_PLAIN, 0.9, (160, 160, 160), 1)

        cv2.line(canvas, (capture_w, 55), (capture_w, h), color, 1)
        btn_w, btn_h, gap = 200, 45, 15
        menu_items = [
            {"id": "TOGGLE_EFF", "label": "EFFECT MODE", "active": (self.mode == "EFFECTOR")},
            {"id": "TOGGLE_SYN", "label": "SYNTH MODE",  "active": (self.mode == "SYNTH")},
            {"id": "GAME_EXIT",  "label": "CLOSE SYSTEM", "active": False}
        ]

        for i, item in enumerate(menu_items):
            bx = capture_w + 20
            by = 80 + i * (btn_h + gap)
            self.buttons_hitboxes[item["id"]] = (bx, by, bx + btn_w, by + btn_h)
            
            bg = color if item["active"] else (25, 25, 25)
            tc = (255, 255, 255) if item["active"] else tuple(int(c*0.6) for c in color)
            cv2.rectangle(canvas, (bx, by), (bx + btn_w, by + btn_h), bg, -1)
            cv2.rectangle(canvas, (bx, by), (bx + btn_w, by + btn_h), color, 1)
            cv2.putText(canvas, item["label"], (bx + 25, by + 28), cv2.FONT_HERSHEY_DUPLEX, 0.5, tc, 1, cv2.LINE_AA)

        # 오디오 파라미터 게이지 바 정보 시각화 및 마우스 클릭 범위 맵핑
        p = self._get_smooth()
        y_off = 95
        for name in self.PARAM_NAMES:
            if name == "Master Vol": continue
            val = p[name]
            bar_w = int(val * 110)
            
            # 텍스트 정보 드로잉
            cv2.putText(canvas, f"{name.upper()}:", (20, y_off), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1)
            cv2.putText(canvas, self.format_param(name, val), (145, y_off), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)
            
            # 마우스 게이지 조작용 힛박스 지정 (가로 110픽셀 영역)
            kx1, ky1, kx2, ky2 = 230, y_off - 12, 340, y_off + 4
            self.knob_hitboxes[name] = (kx1, ky1, kx2, ky2)
            
            # 비주얼 슬라이더 렌더링
            cv2.rectangle(canvas, (kx1, ky1), (kx2, ky2), (30, 30, 30), -1)
            cv2.rectangle(canvas, (kx1, ky1), (kx1 + bar_w, ky2), color, -1)
            y_off += 25

        # 하단 마스터 볼륨 인디케이터 및 마우스 조작 노브 연동
        mv = p["Master Vol"]
        sx, sy = (capture_w // 2) - 100, h - 40
        self.knob_hitboxes["Master Vol"] = (sx, sy, sx + 200, sy + 12)
        
        cv2.rectangle(canvas, (sx, sy), (sx + 200, sy + 12), (30, 30, 30), -1)
        cv2.rectangle(canvas, (sx, sy), (sx + int(mv * 200), sy + 12), color, -1)
        cv2.putText(canvas, f"MASTER VOL (DISTANCE / MOUSE): {int(mv*100)}%", (sx - 60, sy - 8), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1)

    def run(self):
        try:
            dev_info = sd.query_devices(sd.default.device[1])
            self.samplerate = int(dev_info.get('default_samplerate', DEFAULT_SAMPLE_RATE))
        except: pass

        self.filter.samplerate = self.samplerate
        self.delay.samplerate  = self.samplerate
        self.synth.samplerate  = self.samplerate
        self.arp.samplerate    = self.samplerate

        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return

        win_name = "TOUCHLESS_WORKSTATION_v12"
        cv2.namedWindow(win_name)
        # Mouse Callback 시스템 연동 통합
        cv2.setMouseCallback(win_name, self.on_mouse_event)

        self.running = True
        
        # [수정 반영] 맥북 하드웨어 에러 예방을 위해 channels=(1, 2) 튜플 구조 전달 고정
        stream = sd.Stream(
            device=(sd.default.device[0], sd.default.device[1]),
            samplerate=self.samplerate,
            blocksize=BLOCK_SIZE,
            channels=(1, 2),
            callback=self.audio_callback
        )
        stream.start()

        try:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret: continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                canvas = np.zeros_like(frame)

                if self.state == "MAIN_MENU":
                    self.render_main_menu(canvas, w, h)
                elif self.state == "WORKSTATION":
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb)
                    self.render_workstation(frame, canvas, results, w, h)

                cv2.imshow(win_name, canvas)
                key = cv2.waitKey(1) & 0xFF

                if key == 27: self.running = False
                elif self.state == "MAIN_MENU":
                    if key == ord('1'):   self.state = "WORKSTATION"
                    elif key == ord('3'): self.running = False
                elif self.state == "WORKSTATION":
                    if key == 8:          self.state = "MAIN_MENU"
                    elif key == ord('m'): self.mode = "SYNTH" if self.mode == "EFFECTOR" else "EFFECTOR"
                    
                    elif key == ord('n'):
                        keys = list(self.SCALES.keys())
                        idx  = keys.index(self.scale_name)
                        self.scale_name = keys[(idx + 1) % len(keys)]
                        self.arp.scale  = self.SCALES[self.scale_name]
                        print(f"[INFO] Scale -> {self.scale_name}")
                        
                    elif key == ord('s'):
                        p_name = f"preset_{int(time.time())}"
                        self.presets.save(p_name, self._get_smooth())
                        print(f"[INFO] Saved Preset Timestamp Slot: {p_name}")
                        
                    elif ord('1') <= key <= ord('5'):
                        slot = key - ord('1')
                        plist = self.presets.list_presets()
                        if slot < len(plist):
                            loaded = self.presets.load(plist[slot])
                            with self._lock:
                                for k, v in loaded.items():
                                    if k in self._targets: self._targets[k] = float(v)
                            print(f"[INFO] Loaded preset slot data: {plist[slot]}")

        finally:
            self.running = False
            stream.stop()
            stream.close()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    TouchlessWorkstation().run()