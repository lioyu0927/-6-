"""
TOUCHLESS WORKSTATION v12.FULL_RESTORED
─────────────────────────────────────────────────────────────
[긴급 복구: 모든 손가락 관절 트래킹 복원 + 양손 멀티 트레일 및 아스키 UI 완벽 통합]
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

SAMPLE_RATE  = 44100
BLOCK_SIZE   = 512
ALPHA_SMOOTH = 0.04      
PRESET_DIR   = os.path.expanduser("~/.touchless_presets")


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


# ─────────────────────────────────────────────
# DSP COMPONENTS
# ─────────────────────────────────────────────
class BiquadFilter:
    def __init__(self, filter_type="lp"):
        self.filter_type = filter_type
        self._cutoff = 1000.0
        self._q      = 0.707
        self.z1, self.z2 = np.zeros((1, 2)), np.zeros((1, 2))
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
        w0     = 2 * np.pi * self._cutoff / SAMPLE_RATE
        alpha  = np.sin(w0) / (2 * self._q)
        cos_w0 = np.cos(w0)
        if self.filter_type == "lp":
            b0, b1, b2 = (1 - cos_w0) / 2, 1 - cos_w0, (1 - cos_w0) / 2
        else:
            b0, b1, b2 = (1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2
        a0 = 1 + alpha
        self.b0, self.b1, self.b2 = b0 / a0, b1 / a0, b2 / a0
        self.a1, self.a2 = (-2 * cos_w0) / a0, (1 - alpha) / a0
        self._dirty = False

    def process(self, data):
        if self._dirty: self._calculate_coeffs()
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
        d = float(self.drive)
        return np.tanh(data * d) * (1.0 / (1.0 + np.log1p(d * 0.5)))

class Delay:
    def __init__(self, max_delay_sec=1.0):
        self.buffer = np.zeros((int(SAMPLE_RATE * max_delay_sec), 2))
        self.ptr = 0
        self.current_delay_samples = int(SAMPLE_RATE * 0.3)

    def process(self, data, mix=0.3, feedback=0.4):
        out = np.zeros_like(data)
        for i in range(len(data)):
            read_idx = (self.ptr - self.current_delay_samples) % len(self.buffer)
            delayed = self.buffer[read_idx]
            self.buffer[self.ptr] = data[i] + delayed * feedback
            out[i] = data[i] * (1.0 - mix) + delayed * mix
            self.ptr = (self.ptr + 1) % len(self.buffer)
        return out

class SchroederReverb:
    COMB_L, COMB_R = [1557, 1617, 1491, 1422], [1592, 1668, 1525, 1456]
    AP_DELAYS = [225, 341]

    def __init__(self):
        self.size = 0.5
        self._comb_l = [np.zeros(d) for d in self.COMB_L]
        self._comb_r = [np.zeros(d) for d in self.COMB_R]
        self._comb_pl, self._comb_pr = [0]*4, [0]*4
        self._ap_l, self._ap_r = [np.zeros(d) for d in self.AP_DELAYS], [np.zeros(d) for d in self.AP_DELAYS]
        self._ap_pl, self._ap_pr = [0]*2, [0]*2

    def _process_channel(self, mono, comb_bufs, comb_ptrs, ap_bufs, ap_ptrs, decay):
        n = len(mono)
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
                idx = ptr % blen
                delayed = buf[idx]
                buf[idx] = out[i] + delayed * 0.5
                out[i] = delayed - out[i] * 0.5
                ptr = (ptr + 1) % blen
            ap_ptrs[ai] = ptr
        return out

    def process(self, data, mix=0.3):
        if mix < 0.001: return data
        decay = 0.4 + self.size * 0.45
        wet_l = self._process_channel(data[:, 0], self._comb_l, self._comb_pl, self._ap_l, self._ap_pl, decay)
        wet_r = self._process_channel(data[:, 1], self._comb_r, self._comb_pr, self._ap_r, self._ap_pr, decay)
        return data * (1.0 - mix) + np.stack([wet_l, wet_r], axis=1) * mix

class SynthOscillator:
    def __init__(self): self.phase_l, self.phase_r = 0.0, 0.025
    def process(self, num_samples, freq):
        t = np.arange(num_samples)
        phase_l = self.phase_l + 2 * np.pi * freq * (t / SAMPLE_RATE)
        phase_r = self.phase_r + 2 * np.pi * freq * 1.002 * (t / SAMPLE_RATE)
        self.phase_l, self.phase_r = phase_l[-1] % (2 * np.pi), phase_r[-1] % (2 * np.pi)
        return np.stack([
            (0.4 * np.sin(phase_l) + 0.3 * np.sin(2.01 * phase_l)),
            (0.4 * np.sin(phase_r) + 0.3 * np.sin(2.01 * phase_r))
        ], axis=1)

class Arpeggiator:
    def __init__(self, scale):
        self.scale = scale
        self.current_step = 0
        self.timer = 0
        self.current_note = scale[0]

    def update(self, bpm):
        threshold = int(SAMPLE_RATE / (bpm / 60.0 * 8))
        self.timer += BLOCK_SIZE
        if self.timer >= threshold:
            self.timer = 0
            self.current_step = (self.current_step + 1) % len(self.scale)
            self.current_note = self.scale[self.current_step]
        return self.current_note


# ─────────────────────────────────────────────
# AUXILIARY SYSTEMS
# ─────────────────────────────────────────────
class GestureEngine:
    TIP_IDS, JOINT_IDS = [4, 8, 12, 16, 20], [3, 6, 10, 14, 18]
    def classify(self, lms):
        up = [lms[4].x < lms[3].x] + [lms[t].y < lms[j].y for t, j in zip(self.TIP_IDS[1:], self.JOINT_IDS[1:])]
        n = sum(up)
        if n == 0: return "FIST"
        if n == 5: return "OPEN"
        if n == 1 and up[1]: return "POINT"
        if n == 2 and up[1] and up[2]: return "PEACE"
        if np.hypot(lms[8].x - lms[4].x, lms[8].y - lms[4].y) < 0.05: return "PINCH"
        return "FREE"

class HandTrail:
    def __init__(self, color=(0, 255, 65)):
        self.points = deque(maxlen=28)
        self.color = color
    def add(self, pt): self.points.append(pt)
    def draw(self, canvas):
        for i, pt in enumerate(self.points):
            alpha = (i + 1) / len(self.points)
            cv2.circle(canvas, pt, max(1, int(alpha * 5)), tuple(int(ch * alpha) for ch in self.color), -1)

class PresetManager:
    def __init__(self):
        os.makedirs(PRESET_DIR, exist_ok=True)
        self.slots = {}
        self.load_all()
    def load_all(self):
        for f in os.listdir(PRESET_DIR):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(PRESET_DIR, f)) as fh: self.slots[f[:-5]] = json.load(fh)
                except: pass
    def save(self, name, params):
        self.slots[name] = dict(params)
        with open(os.path.join(PRESET_DIR, f"{name}.json"), "w") as fh: json.dump(params, fh, indent=2)
    def list_presets(self): return sorted(self.slots.keys())


# ─────────────────────────────────────────────
# MAIN CONTROLLER
# ─────────────────────────────────────────────
class TouchlessWorkstation:
    SCALES = {
        "PENTA": [261.63, 293.66, 349.23, 392.00, 440.00, 523.25],
        "MINOR": [261.63, 293.66, 311.13, 349.23, 392.00, 415.30, 466.16, 523.25],
    }
    _PARAM_DEFAULTS = {
        "Master Vol": 0.7, "Filter Cutoff": 0.5, "Filter Res": 0.2,
        "Distortion": 0.0, "Glitch": 0.0, "Delay Mix": 0.0,
        "Reverb Mix": 0.0, "Synth Vol": 0.5, "Arp BPM": 0.4
    }

    def __init__(self):
        self.app_state = "MAIN_MENU"  
        self.mode = "EFFECTOR"
        self.scale_name = "PENTA"
        self.running = False
        self.selected_preset_idx = 0

        self._lock = threading.Lock()
        self._smooth = {k: SmoothParam(v) for k, v in self._PARAM_DEFAULTS.items()}
        self._targets = dict(self._PARAM_DEFAULTS)

        self.filter, self.dist, self.delay, self.reverb = BiquadFilter(), Distortion(), Delay(), SchroederReverb()
        self.synth, self.arp = SynthOscillator(), Arpeggiator(self.SCALES[self.scale_name])
        self.gesture_engine = GestureEngine()
        self.presets = PresetManager()
        self.trails = [HandTrail((0, 255, 65)), HandTrail((200, 50, 255))]

        self._hold_start = {}
        self.ascii_chars = " .:-=+*#%@"
        self.ascii_scale = 10

        self.mouse_hitboxes = {}
        self.is_dragging = None

    def _set_target(self, key, val):
        with self._lock: self._targets[key] = float(np.clip(val, 0.0, 1.0))

    def _get_smooth(self):
        with self._lock: return {k: sp.value for k, sp in self._smooth.items()}

    def _check_hold(self, gesture: str, threshold: float) -> bool:
        now = time.time()
        if gesture not in self._hold_start:
            self._hold_start[gesture] = now
            return False
        return (now - self._hold_start[gesture]) >= threshold

    def _clear_hold(self, gesture: str): self._hold_start.pop(gesture, None)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.app_state == "MAIN_MENU":
                if "START_BTN" in self.mouse_hitboxes:
                    x1, y1, x2, y2 = self.mouse_hitboxes["START_BTN"]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.app_state = "WORKSTATION"
                        return
                if "EXIT_BTN" in self.mouse_hitboxes:
                    x1, y1, x2, y2 = self.mouse_hitboxes["EXIT_BTN"]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.running = False
                        return
                for i in range(5):
                    box_name = f"PRESET_SLOT_{i}"
                    if box_name in self.mouse_hitboxes:
                        x1, y1, x2, y2 = self.mouse_hitboxes[box_name]
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            self.selected_preset_idx = i
                            plist = self.presets.list_presets()
                            if i < len(plist):
                                loaded = self.presets.slots[plist[i]]
                                with self._lock:
                                    for k, v in loaded.items(): self._targets[k] = float(v)
                            return

            elif self.app_state == "WORKSTATION":
                if "BACK_BTN" in self.mouse_hitboxes:
                    x1, y1, x2, y2 = self.mouse_hitboxes["BACK_BTN"]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.app_state = "MAIN_MENU"
                        return

                for name, (x1, y1, x2, y2) in self.mouse_hitboxes.items():
                    if name in self._targets and x1 <= x <= x2 and y1 <= y <= y2:
                        self.is_dragging = name
                        self._set_target(name, (x - x1) / (x2 - x1))
                        break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.app_state == "WORKSTATION" and self.is_dragging:
                x1, _, x2, _ = self.mouse_hitboxes[self.is_dragging]
                self._set_target(self.is_dragging, (x - x1) / (x2 - x1))

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = None

    def audio_callback(self, indata, outdata, frames, time_info, status):
        with self._lock:
            for k, sp in self._smooth.items(): sp.set_target(self._targets[k])
        p = self._get_smooth()

        if self.mode == "EFFECTOR" and indata is not None:
            sig = indata.copy() if indata.shape[1] >= 2 else np.repeat(indata, 2, axis=1)
        else:
            sig = self.synth.process(frames, self.arp.update(60 + p["Arp BPM"] * 180)) * p["Synth Vol"]

        self.dist.drive = 1.0 + p["Distortion"] * 25.0
        sig = self.dist.process(sig)

        self.filter.cutoff = 20 * (1000 ** p["Filter Cutoff"])
        self.filter.q = 0.1 + p["Filter Res"] * 4.9
        sig = self.filter.process(sig)

        sig = self.delay.process(sig, mix=p["Delay Mix"], feedback=p["Delay Mix"] * 0.7)
        self.reverb.size = p["Reverb Mix"]
        sig = self.reverb.process(sig, mix=p["Reverb Mix"])

        outdata[:] = np.clip(sig * p["Master Vol"], -1.0, 1.0)

    def render_ascii(self, frame: np.ndarray) -> np.ndarray:
        h, w, _ = frame.shape
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (w // self.ascii_scale, h // self.ascii_scale))
        small = cv2.equalizeHist(small)
        canvas = np.zeros_like(frame)
        p = self._get_smooth()
        color = (0, int(p["Filter Cutoff"] * 180 + 60), int(p["Distortion"] * 220))
        cw, ch = self.ascii_scale, self.ascii_scale
        n = len(self.ascii_chars)

        for i in range(small.shape[0]):
            for j in range(small.shape[1]):
                idx = min(int(small[i, j] / 256 * n), n - 1)
                cv2.putText(canvas, self.ascii_chars[idx], (j * cw, i * ch + ch), cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1)
        return canvas

    def draw_main_menu(self, canvas):
        self.mouse_hitboxes.clear()
        cv2.putText(canvas, "TOUCHLESS WORKSTATION v12", (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 65), 2)
        cv2.putText(canvas, "MAIN SYSTEM MENU", (50, 120), cv2.FONT_HERSHEY_PLAIN, 1.2, (180, 180, 180), 1)
        cv2.line(canvas, (50, 140), (640 - 50, 140), (0, 255, 65), 1)

        bx1, by1, bx2, by2 = 50, 180, 250, 240
        self.mouse_hitboxes["START_BTN"] = (bx1, by1, bx2, by2)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 40, 10), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 65), 2)
        cv2.putText(canvas, "LAUNCH ENGINE", (70, 218), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 255, 65), 2)

        ex1, ey1, ex2, ey2 = 50, 260, 250, 320
        self.mouse_hitboxes["EXIT_BTN"] = (ex1, ey1, ex2, ey2)
        cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), (10, 10, 40), -1)
        cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), (0, 100, 255), 1)
        cv2.putText(canvas, "EXIT SYSTEM", (85, 298), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 100, 255), 1)

        cv2.putText(canvas, "LOAD SYSTEM PRESET SLOTS:", (320, 180), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        plist = self.presets.list_presets()
        for i in range(5):
            px1, py1, px2, py2 = 320, 200 + (i * 45), 600, 235 + (i * 45)
            self.mouse_hitboxes[f"PRESET_SLOT_{i}"] = (px1, py1, px2, py2)
            is_sel = (self.selected_preset_idx == i)
            bg_col = (0, 70, 20) if is_sel else (30, 30, 30)
            brd_col = (0, 255, 65) if is_sel else (100, 100, 100)
            cv2.rectangle(canvas, (px1, py1), (px2, py2), bg_col, -1)
            cv2.rectangle(canvas, (px1, py1), (px2, py2), brd_col, 1)
            ptext = f" Slot {i+1}: {plist[i][-15:]}" if i < len(plist) else f" Slot {i+1}: [EMPTY_SLOT]"
            cv2.putText(canvas, ptext, (330, 222 + (i * 45)), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)

    def draw_workstation_ui(self, frame, w, h):
        p = self._get_smooth()
        color = (0, int(p["Filter Cutoff"] * 180 + 60), int(p["Distortion"] * 220))

        self.mouse_hitboxes["BACK_BTN"] = (20, 20, 110, 50)
        cv2.rectangle(frame, (20, 20), (110, 50), (20, 20, 20), -1)
        cv2.rectangle(frame, (20, 20), (110, 50), color, 1)
        cv2.putText(frame, "< MENU", (35, 40), cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)

        banner = f"// ENGINE: {self.mode}"
        cv2.putText(frame, banner, (135, 42), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)

        y_off = 98
        for name, sp in self._smooth.items():
            if name == "Master Vol": continue
            val = sp.value
            cv2.putText(frame, f"{name.upper()}:", (28, y_off), cv2.FONT_HERSHEY_PLAIN, 0.85, color, 1)
            
            kx1, ky1, kx2, ky2 = 220, y_off - 11, 220 + 150, y_off + 1
            self.mouse_hitboxes[name] = (kx1, ky1, kx2, ky2)
            cv2.rectangle(frame, (kx1, ky1), (kx2, ky2), (35, 35, 35), -1)
            cv2.rectangle(frame, (kx1, ky1), (kx1 + int(val * 150), ky2), color, -1)
            y_off += 32

        mv = p["Master Vol"]
        sx, sy = w // 2 - 150, h - 60
        self.mouse_hitboxes["Master Vol"] = (sx, sy, sx + 300, sy + 15)
        cv2.rectangle(frame, (sx, sy), (sx + 300, sy + 15), (35, 35, 35), -1)
        cv2.rectangle(frame, (sx, sy), (sx + int(mv * 300), sy + 15), color, -1)
        cv2.putText(frame, f"MASTER VOLUME: {int(mv*100)}%", (sx, h - 68), cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)

    def run(self):
        win_name = "TOUCHLESS WORKSTATION"
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, self.on_mouse)

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75)
        cap = cv2.VideoCapture(0)

        devices = sd.query_devices()
        in_idx, out_idx = sd.default.device[0], sd.default.device[1]
        for i, d in enumerate(devices):
            if "blackhole" in d['name'].lower(): in_idx = i
            if "built-in" in d['name'].lower() or "다중" in d['name'].lower(): out_idx = i

        self.running = True
        try:
            with sd.Stream(device=(in_idx, out_idx), samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=(1,2), callback=self.audio_callback):
                while self.running:
                    ret, frame = cap.read()
                    if not ret: frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    frame = cv2.flip(frame, 1)

                    if self.app_state == "MAIN_MENU":
                        menu_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                        self.draw_main_menu(menu_canvas)
                        cv2.imshow(win_name, menu_canvas)
                    
                    elif self.app_state == "WORKSTATION":
                        canvas = self.render_ascii(frame)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = hands.process(rgb)
                        
                        if results.multi_hand_landmarks:
                            gestures = [self.gesture_engine.classify(lm.landmark) for lm in results.multi_hand_landmarks]
                            
                            if any(g == "PEACE" for g in gestures):
                                if self._check_hold("PEACE", 0.5):
                                    self.mode = "SYNTH" if self.mode == "EFFECTOR" else "EFFECTOR"
                                    self._clear_hold("PEACE")
                            else:
                                self._clear_hold("PEACE")
                            
                            # [핵심 복구] 양손 트레일 궤적과 다섯 손가락 끝 마커 완벽 통합 그리기
                            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                                lms = hand_lms.landmark
                                
                                # 검지 손가락 끝(8) 기준 양손 트레일 포인트 추가 및 렌더링
                                ix, iy = int(lms[8].x * frame.shape[1]), int(lms[8].y * frame.shape[0])
                                self.trails[idx % 2].add((ix, iy))
                                self.trails[idx % 2].draw(canvas)
                                
                                # 다섯 손가락 모든 끝 단 관절(엄지4, 검지8, 중지12, 약지16, 새끼20) 좌표 인식 서클 그리기
                                for p_id in [4, 8, 12, 16, 20]:
                                    fx = int(lms[p_id].x * frame.shape[1])
                                    fy = int(lms[p_id].y * frame.shape[0])
                                    cv2.circle(canvas, (fx, fy), 4, (255, 255, 255), -1)

                        self.draw_workstation_ui(canvas, frame.shape[1], frame.shape[0])
                        cv2.imshow(win_name, canvas)

                    if cv2.waitKey(1) & 0xFF == 27: break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    TouchlessWorkstation().run()