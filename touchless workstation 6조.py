

import os
import json
import time
import platform
import threading
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DEFAULT_SAMPLE_RATE = 44100
SAMPLE_RATE         = DEFAULT_SAMPLE_RATE   # legacy alias
BLOCK_SIZE          = 512
CHANNELS            = 2
ALPHA_SMOOTH        = 0.04
TRAIL_LENGTH        = 28
PRESET_DIR          = os.path.expanduser("~/.touchless_presets")

# 레이아웃 상수 (run()에서 최종 덮어씀)
SIDEBAR_W   = 260
TOPBAR_H    = 50
BOTTOMBAR_H = 50

# ── UI 색상 팔레트
UI_BG_DARK      = (18, 18, 28)
UI_PANEL_BG     = (28, 28, 42)
UI_ACCENT_CYAN  = (0, 230, 255)
UI_ACCENT_PINK  = (255, 80, 200)
UI_ACCENT_LIME  = (80, 255, 100)
UI_ACCENT_AMBER = (255, 190, 30)
UI_TEXT_WHITE   = (240, 240, 255)
UI_TEXT_DIM     = (120, 120, 150)
UI_LEFT_ZONE    = (255, 180, 50)
UI_RIGHT_ZONE   = (50, 220, 255)

# ── 제스처 레이블 (한국어)
GESTURE_LABELS = {
    "FIST":  "주먹",
    "OPEN":  "펼침",
    "POINT": "검지",
    "PEACE": "브이",
    "PINCH": "핀치",
    "FREE":  "자유",
}

# ── 제스처별 색상
GESTURE_COLORS = {
    "FIST":  (255, 80,  80),
    "OPEN":  (80,  255, 80),
    "POINT": (80,  180, 255),
    "PEACE": (255, 200, 50),
    "PINCH": (255, 80,  200),
    "FREE":  (160, 160, 180),
}

# ── 온보딩 가이드
GESTURE_GUIDE = {
    "EFFECTOR": [
        ("왼손 핀치",    "딜레이 Mix / 리버브"),
        ("왼손 주먹",    "Arp BPM / 신스 볼륨"),
        ("오른손 핀치",  "필터 공명 / 컷오프"),
        ("오른손 주먹",  "디스토션 / 글리치"),
        ("양손 펼침 1초","모든 값 초기화"),
        ("브이 0.5초",   "모드 전환"),
    ],
    "SYNTH": [
        ("왼손 핀치",    "딜레이 Mix / 리버브"),
        ("왼손 주먹",    "Arp BPM / 신스 볼륨"),
        ("오른손 핀치",  "필터 공명 / 컷오프"),
        ("오른손 주먹",  "디스토션 / 글리치"),
        ("양손 펼침 1초","모든 값 초기화"),
        ("브이 0.5초",   "모드 전환"),
    ],
}

KEY_GUIDE = [
    ("M",   "모드 전환"),
    ("N",   "음계 변경"),
    ("S",   "프리셋 저장"),
    ("1-5", "프리셋 불러오기"),
    ("G",   "가이드 토글"),
    ("ESC", "종료"),
]


# ─────────────────────────────────────────────
# SMOOTH PARAMETER
# ─────────────────────────────────────────────
class SmoothParam:
    def __init__(self, init=0.0, alpha=ALPHA_SMOOTH):
        self.value   = float(init)
        self.alpha   = alpha
        self._target = float(init)

    def set_target(self, target):
        self._target = float(target)
        self.value  += self.alpha * (self._target - self.value)
        return self.value

    def tick(self):
        self.value += self.alpha * (self._target - self.value)
        return self.value

    def __float__(self): return self.value


# ─────────────────────────────────────────────
# DSP COMPONENTS
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
            self._cutoff = v; self._dirty = True

    @property
    def q(self): return self._q
    @q.setter
    def q(self, v):
        v = float(np.clip(v, 0.1, 5.0))
        if abs(v - self._q) > 0.001:
            self._q = v; self._dirty = True

    def _calculate_coeffs(self):
        w0     = 2 * np.pi * self._cutoff / self.samplerate
        alpha  = np.sin(w0) / (2 * self._q)
        cos_w0 = np.cos(w0)
        if self.filter_type == "lp":
            b0 = (1 - cos_w0) / 2; b1 = 1 - cos_w0; b2 = (1 - cos_w0) / 2
        else:
            b0 = (1 + cos_w0) / 2; b1 = -(1 + cos_w0); b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        self.b0 = b0/a0; self.b1 = b1/a0; self.b2 = b2/a0
        self.a1 = (-2*cos_w0)/a0; self.a2 = (1-alpha)/a0
        self._dirty = False

    def process(self, data):
        if self._dirty: self._calculate_coeffs()
        out = np.zeros_like(data)
        for i in range(len(data)):
            x = data[i:i+1]
            y = self.b0*x + self.z1
            self.z1 = self.b1*x - self.a1*y + self.z2
            self.z2 = self.b2*x - self.a2*y
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
        self.buffer = np.zeros((int(samplerate * max_delay_sec), CHANNELS))
        self.ptr = 0
        self.current_delay_samples = int(samplerate * 0.3)

    def process(self, data, mix=0.3, feedback=0.4):
        out = np.zeros_like(data)
        for i in range(len(data)):
            read_idx = (self.ptr - self.current_delay_samples) % len(self.buffer)
            delayed  = self.buffer[read_idx]
            self.buffer[self.ptr] = data[i] + delayed * feedback
            out[i]   = data[i] * (1.0 - mix) + delayed * mix
            self.ptr = (self.ptr + 1) % len(self.buffer)
        return out


class SchroederReverb:
    COMB_L    = [1557, 1617, 1491, 1422]
    COMB_R    = [1592, 1668, 1525, 1456]
    AP_DELAYS = [225, 341]

    def __init__(self):
        self.size = 0.5
        self._comb_l  = [np.zeros(d) for d in self.COMB_L]
        self._comb_r  = [np.zeros(d) for d in self.COMB_R]
        self._comb_pl = [0]*4; self._comb_pr = [0]*4
        self._ap_l    = [np.zeros(d) for d in self.AP_DELAYS]
        self._ap_r    = [np.zeros(d) for d in self.AP_DELAYS]
        self._ap_pl   = [0]*2;  self._ap_pr = [0]*2

    def _process_channel(self, mono, comb_bufs, comb_ptrs, ap_bufs, ap_ptrs, decay):
        n = len(mono); out = np.zeros(n)
        for ci, (buf, ptr) in enumerate(zip(comb_bufs, comb_ptrs)):
            blen = len(buf)
            for i in range(n):
                idx = ptr % blen; delayed = buf[idx]
                buf[idx] = mono[i] + delayed * decay
                out[i] += delayed; ptr = (ptr + 1) % blen
            comb_ptrs[ci] = ptr
        out /= 4.0
        for ai, (buf, ptr) in enumerate(zip(ap_bufs, ap_ptrs)):
            blen = len(buf)
            for i in range(n):
                idx = ptr % blen; delayed = buf[idx]
                buf[idx] = out[i] + delayed * 0.5
                out[i]   = delayed - out[i] * 0.5
                ptr = (ptr + 1) % blen
            ap_ptrs[ai] = ptr
        return out

    def process(self, data, mix=0.3):
        if mix < 0.001: return data
        decay = 0.4 + self.size * 0.45
        wet_l = self._process_channel(data[:,0], self._comb_l, self._comb_pl, self._ap_l, self._ap_pl, decay)
        wet_r = self._process_channel(data[:,1], self._comb_r, self._comb_pr, self._ap_r, self._ap_pr, decay)
        return data * (1.0 - mix) + np.stack([wet_l, wet_r], axis=1) * mix


class SynthOscillator:
    """3-harmonic stereo oscillator (b.py 개선판)."""
    def __init__(self, samplerate=DEFAULT_SAMPLE_RATE):
        self.samplerate = samplerate
        self.phase_l = 0.0; self.phase_r = 0.025

    def process(self, num_samples, freq):
        t  = np.arange(num_samples)
        pl = self.phase_l + 2*np.pi*freq*(t/self.samplerate)
        pr = self.phase_r + 2*np.pi*freq*1.002*(t/self.samplerate)
        self.phase_l = pl[-1] % (2*np.pi)
        self.phase_r = pr[-1] % (2*np.pi)
        sl = 0.4*np.sin(pl) + 0.3*np.sin(2.01*pl) + 0.2*np.sin(3.02*pl)
        sr = 0.4*np.sin(pr) + 0.3*np.sin(2.01*pr) + 0.2*np.sin(3.02*pr)
        return np.stack([sl, sr], axis=1)


class Arpeggiator:
    def __init__(self, scale, samplerate=DEFAULT_SAMPLE_RATE):
        self.scale = scale; self.samplerate = samplerate
        self.current_step = 0; self.timer = 0
        self.current_note = scale[0]

    def update(self, bpm):
        threshold = int(self.samplerate / (bpm / 60.0 * 8))
        self.timer += BLOCK_SIZE
        if self.timer >= threshold:
            self.timer = 0
            self.current_step = (self.current_step + 1) % len(self.scale)
            self.current_note = self.scale[self.current_step]
        return self.current_note


# ─────────────────────────────────────────────
# GESTURE ENGINE
# ─────────────────────────────────────────────
class GestureEngine:
    TIP_IDS   = [4, 8, 12, 16, 20]
    JOINT_IDS = [3, 6, 10, 14, 18]

    def count_fingers(self, lms):
        up = [lms[4].x < lms[3].x]
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


# ─────────────────────────────────────────────
# HAND TRAIL
# ─────────────────────────────────────────────
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
            radius = max(1, int(alpha * 6))
            c = tuple(int(ch * alpha) for ch in self.color)
            cv2.circle(canvas, pt, radius, c, -1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# PRESET MANAGER
# ─────────────────────────────────────────────
class PresetManager:
    def __init__(self, directory=PRESET_DIR):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.slots: dict = {}
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
    def list_presets(self) -> list: return sorted(self.slots.keys())


# ─────────────────────────────────────────────
# UI HELPER (b.py – 한국어 폰트·둥근 사각형·글로우 등)
# ─────────────────────────────────────────────
class UIHelper:
    """재사용 가능한 UI 드로잉 유틸리티."""

    _fonts: dict = {}
    _batch_pil  = None
    _batch_draw = None

    @classmethod
    def _font_paths(cls):
        sys = platform.system()
        if sys == "Windows":
            fd = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
            return [os.path.join(fd, f) for f in ("malgun.ttf", "malgunbd.ttf", "gulim.ttc")]
        if sys == "Darwin":
            return [
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
                "/Library/Fonts/Arial Unicode.ttf",
            ]
        return [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

    @classmethod
    def _get_font(cls, size: int):
        key = ("main", size)
        if key in cls._fonts: return cls._fonts[key]
        for path in cls._font_paths():
            if not os.path.isfile(path): continue
            for kwargs in ({}, {"index": 0}):
                try:
                    font = ImageFont.truetype(path, size, **kwargs)
                    cls._fonts[key] = font; return font
                except (OSError, IOError): continue
        font = ImageFont.load_default()
        cls._fonts[key] = font; return font

    @classmethod
    def text_size(cls, text: str, size: int = 16):
        font = cls._get_font(size)
        bbox = font.getbbox(text or " ")
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    @classmethod
    def text_begin(cls, img):
        cls._batch_pil  = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cls._batch_draw = ImageDraw.Draw(cls._batch_pil)

    @classmethod
    def text_end(cls, img):
        if cls._batch_pil is not None:
            img[:] = cv2.cvtColor(np.array(cls._batch_pil), cv2.COLOR_RGB2BGR)
        cls._batch_pil = None; cls._batch_draw = None

    @classmethod
    def put_text(cls, img, text, xy, size=16, color=(255, 255, 255), anchor="ls"):
        if not text: return
        font = cls._get_font(size)
        fill = (int(color[2]), int(color[1]), int(color[0]))
        if cls._batch_draw is not None:
            cls._batch_draw.text(xy, text, font=font, fill=fill, anchor=anchor); return
        pil  = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text(xy, text, font=font, fill=fill, anchor=anchor)
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    @classmethod
    def put_text_cv(cls, img, text, xy, size=16, color=(255, 255, 255)):
        cls.put_text(img, text, (xy[0], xy[1]), size=size, color=color, anchor="ls")

    @staticmethod
    def rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1, alpha=1.0):
        x1, y1 = pt1; x2, y2 = pt2
        if x2 <= x1 or y2 <= y1: return
        overlay = img.copy()
        r = min(radius, (x2-x1)//2, (y2-y1)//2)
        cv2.rectangle(overlay, (x1+r, y1), (x2-r, y2), color, thickness)
        cv2.rectangle(overlay, (x1, y1+r), (x2, y2-r), color, thickness)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(overlay, (cx, cy), r, color, thickness)
        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
        else:
            img[:] = overlay

    @staticmethod
    def glow_circle(img, center, radius, color, intensity=0.5):
        overlay = img.copy()
        for r_off, _ in [(radius+8, 0.08), (radius+4, 0.15), (radius, 0.9)]:
            cv2.circle(overlay, center, r_off, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, intensity, img, 1-intensity, 0, img)

    @classmethod
    def text_center(cls, img, text, cx, cy, size=16, color=(255, 255, 255)):
        tw, th = cls.text_size(text, size)
        cls.put_text(img, text, (cx - tw//2, cy + th//2), size=size, color=color, anchor="ls")

    @staticmethod
    def draw_bar(img, x, y, w, h, value, fg_color, bg_color=(40, 40, 55), radius=3):
        UIHelper.rounded_rect(img, (x, y), (x+w, y+h), bg_color, radius)
        fill_w = max(radius*2, int(value * w))
        UIHelper.rounded_rect(img, (x, y), (x+fill_w, y+h), fg_color, radius)

    @staticmethod
    def panel(img, pt1, pt2, alpha=0.65, radius=12):
        x1, y1 = pt1; x2, y2 = pt2
        overlay = img.copy()
        UIHelper.rounded_rect(overlay, pt1, pt2, UI_PANEL_BG, radius, -1)
        UIHelper.rounded_rect(overlay, pt1, pt2, (60, 60, 80), radius, 1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)


# ─────────────────────────────────────────────
# MAIN CONTROLLER
# ─────────────────────────────────────────────
class TouchlessWorkstation:
    # ── 음계 (b.py: BLUES / MAJOR 추가)
    SCALES = {
        "PENTA": [261.63, 293.66, 349.23, 392.00, 440.00, 523.25],
        "MINOR": [261.63, 293.66, 311.13, 349.23, 392.00, 415.30, 466.16, 523.25],
        "BLUES": [261.63, 311.13, 349.23, 369.99, 392.00, 466.16, 523.25],
        "MAJOR": [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25],
    }
    SCALE_KR = {"PENTA": "펜타토닉", "MINOR": "마이너", "BLUES": "블루스", "MAJOR": "메이저"}

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

    _PARAM_META = {
        "Filter Cutoff": ("필터 컷오프", "~", UI_ACCENT_CYAN),
        "Filter Res":    ("필터 공명",   "Q", UI_ACCENT_CYAN),
        "Distortion":    ("디스토션",    "D", (255, 80, 80)),
        "Glitch":        ("글리치",      "G", (200, 80, 255)),
        "Delay Mix":     ("딜레이",      "⟲", UI_ACCENT_LIME),
        "Reverb Mix":    ("리버브",      "◎", UI_ACCENT_LIME),
        "Synth Vol":     ("신스 볼륨",   "♩", UI_ACCENT_PINK),
        "Arp BPM":       ("아르페지오",  "♪", UI_ACCENT_PINK),
    }

    def __init__(self):
        # ── jw10: 메인메뉴 상태 머신
        self.app_state  = "MAIN_MENU"
        self.mode       = "EFFECTOR"
        self.scale_name = "PENTA"
        self.running    = False
        self.selected_preset_idx = 0

        # ── b.py 추가 상태
        self.params_locked  = False
        self.precision_mode = False
        self.samplerate     = DEFAULT_SAMPLE_RATE
        self._show_guide    = True
        self._fps_buf       = deque(maxlen=30)
        self._prev_time     = time.time()
        self._last_gestures = {}

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
        self.hands    = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.6,
        )
        self.gesture_engine = GestureEngine()
        self.trails = [
            HandTrail(color=UI_LEFT_ZONE),
            HandTrail(color=UI_RIGHT_ZONE),
        ]

        self._hold_start: dict = {}
        self.presets = PresetManager()
        self.ascii_chars = " .:-=+*#%@"
        self.ascii_scale = 10

        # ── 마우스 히트박스 (jw10 메인메뉴 + b.py 워크스테이션 버튼)
        self.mouse_hitboxes:   dict = {}   # 메인메뉴용
        self.buttons_hitboxes: dict = {}   # 워크스테이션 사이드바용
        self.is_dragging = None

    # ─────────────────────────────────────────────
    # 파라미터 제어
    # ─────────────────────────────────────────────
    def _set_target(self, key: str, val: float):
        if self.params_locked: return
        with self._lock:
            mult = 0.2 if self.precision_mode else 1.0
            cur  = self._targets[key]
            self._targets[key] = float(np.clip(cur + (float(val) - cur) * mult, 0.0, 1.0))

    def _set_target_abs(self, key: str, val: float):
        """마우스 드래그 등 절댓값 직접 설정용 (precision 무시)."""
        with self._lock:
            self._targets[key] = float(np.clip(val, 0.0, 1.0))

    def _tick_smooth(self):
        with self._lock:
            for k, sp in self._smooth.items(): sp.set_target(self._targets[k])

    def _get_smooth(self) -> dict:
        with self._lock: return {k: sp.value for k, sp in self._smooth.items()}

    def _reset_params(self):
        with self._lock: self._targets = dict(self._PARAM_DEFAULTS)

    @staticmethod
    def format_param(name: str, val: float) -> str:
        if   name == "Filter Cutoff": return f"{20*(1000**val):.0f} Hz"
        elif name == "Filter Res":    return f"Q {0.1+val*4.9:.2f}"
        elif name == "Distortion":    return f"{val*28:.1f} dB"
        elif name == "Glitch":        return f"{val*100:.0f}%"
        elif name == "Delay Mix":     return f"{val*100:.0f}%"
        elif name == "Reverb Mix":    return f"{val*100:.0f}%"
        elif name == "Synth Vol":     return f"{val*100:.0f}%"
        elif name == "Arp BPM":       return f"{60+val*180:.0f} BPM"
        elif name == "Master Vol":    return f"{val*100:.0f}%"
        return f"{val:.2f}"

    def _reactive_color(self) -> tuple:
        p = self._get_smooth()
        r = int(p["Distortion"] * 220)
        g = int(p["Filter Cutoff"] * 180 + 60)
        b = int(p["Reverb Mix"] * 180)
        if self.mode == "SYNTH": return (min(b+120, 255), 20, min(r+100, 255))
        return (0, g, r)

    # ─────────────────────────────────────────────
    # 오디오 콜백
    # ─────────────────────────────────────────────
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
        sig = self.delay.process(sig, mix=p["Delay Mix"], feedback=p["Delay Mix"]*0.7)
        self.reverb.size = p["Reverb Mix"]
        sig = self.reverb.process(sig, mix=p["Reverb Mix"])
        outdata[:] = np.clip(sig * p["Master Vol"], -1.0, 1.0)

    # ─────────────────────────────────────────────
    # ASCII 렌더 (Glitch 효과 포함 – b.py)
    # ─────────────────────────────────────────────
    def render_ascii(self, frame: np.ndarray) -> np.ndarray:
        h, w, _ = frame.shape
        capture_w = w - SIDEBAR_W
        cropped = frame[:, :capture_w]
        small   = cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY),
                             (capture_w // self.ascii_scale, h // self.ascii_scale))
        small   = cv2.equalizeHist(small)
        canvas  = np.zeros_like(frame)
        color   = self._reactive_color()
        cw, ch  = self.ascii_scale, self.ascii_scale
        n       = len(self.ascii_chars)
        glitch  = self._smooth["Glitch"].value
        rows, cols = small.shape

        for i in range(rows):
            shift = 0
            if glitch > 0.05 and np.random.random() < glitch * 0.6:
                shift = int(np.random.uniform(-glitch*8, glitch*8))
            row_color = color
            if glitch > 0.2 and np.random.random() < glitch * 0.25:
                row_color = tuple(255 - c for c in color)
            for j in range(cols):
                src_j = min(max(j+shift, 0), cols-1)
                idx   = min(int(small[i, src_j] / 256 * n), n-1)
                if glitch > 0.4 and np.random.random() < glitch*0.15:
                    idx = np.random.randint(0, n)
                cv2.putText(canvas, self.ascii_chars[idx],
                            (j*cw, i*ch+ch), cv2.FONT_HERSHEY_PLAIN, 0.7, row_color, 1)

        if glitch > 0.5:
            for _ in range(int(glitch*6)):
                by = np.random.randint(0, h)
                bh = np.random.randint(2, 8)
                canvas[by:by+bh, :capture_w] = np.clip(
                    canvas[by:by+bh, :capture_w].astype(int)+60, 0, 255).astype(np.uint8)
        return canvas

    # ─────────────────────────────────────────────
    # 홀드 타이머
    # ─────────────────────────────────────────────
    def _check_hold(self, gesture: str, threshold: float) -> bool:
        now = time.time()
        if gesture not in self._hold_start:
            self._hold_start[gesture] = now; return False
        return (now - self._hold_start[gesture]) >= threshold

    def _clear_hold(self, gesture: str): self._hold_start.pop(gesture, None)

    # ─────────────────────────────────────────────
    # 제어 존 계산
    # ─────────────────────────────────────────────
    def _get_control_zones(self, capture_w: int, h: int):
        ui_top   = TOPBAR_H
        ui_bot   = BOTTOMBAR_H
        usable_h = h - ui_top - ui_bot
        half_w   = capture_w // 2
        zone_w   = int(half_w * 0.70)
        zone_h   = int(usable_h * 0.70)
        cy       = ui_top + usable_h // 2

        left_cx  = half_w // 2
        right_cx = half_w + half_w // 2

        left_zone  = (left_cx - zone_w//2, cy - zone_h//2, left_cx + zone_w//2, cy + zone_h//2)
        right_zone = (right_cx - zone_w//2, cy - zone_h//2, right_cx + zone_w//2, cy + zone_h//2)
        return left_zone, right_zone

    @staticmethod
    def _point_in_zone(x, y, zone):
        x1, y1, x2, y2 = zone
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def _norm_in_zone(x, y, zone):
        x1, y1, x2, y2 = zone
        nx = float(np.clip((x-x1) / max(x2-x1, 1), 0.0, 1.0))
        ny = float(np.clip(1.0-(y-y1) / max(y2-y1, 1), 0.0, 1.0))
        return nx, ny

    # ─────────────────────────────────────────────
    # 제어 존 드로잉 (b.py)
    # ─────────────────────────────────────────────
    def _draw_control_zone(self, canvas, zone, title, subtitle, base_color, active, current_gesture=None):
        x1, y1, x2, y2 = zone

        overlay    = canvas.copy()
        fill_color = tuple(min(c+30, 255) for c in base_color)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
        cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)

        border_color = base_color if active else tuple(int(c*0.7) for c in base_color)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 5, cv2.LINE_AA)

        # 모서리 강조
        corner = 40
        for (bx, by), dx, dy in [((x1,y1),1,1),((x2,y1),-1,1),((x1,y2),1,-1),((x2,y2),-1,-1)]:
            cv2.line(canvas, (bx, by), (bx+dx*corner, by), border_color, 6)
            cv2.line(canvas, (bx, by), (bx, by+dy*corner), border_color, 6)

        # 중앙 십자선
        cx = (x1+x2)//2; cy = (y1+y2)//2
        cv2.line(canvas, (cx, y1), (cx, y2), border_color, 1)
        cv2.line(canvas, (x1, cy), (x2, cy), border_color, 1)

        UIHelper.put_text_cv(canvas, title,    (x1+15, y1+25), size=18, color=border_color)
        UIHelper.put_text_cv(canvas, subtitle, (x1+15, y1+45), size=14, color=(220, 220, 220))

        if active and current_gesture:
            g_label = GESTURE_LABELS.get(current_gesture, current_gesture)
            g_color = GESTURE_COLORS.get(current_gesture, border_color)
            cx_zone = (x1+x2)//2; cy_zone = y2-22
            tw, _   = UIHelper.text_size(g_label, 16)
            UIHelper.rounded_rect(canvas, (cx_zone-tw//2-10, cy_zone-14),
                                  (cx_zone+tw//2+10, cy_zone+6),
                                  tuple(c//3 for c in g_color), 8, -1, 0.85)
            UIHelper.text_center(canvas, g_label, cx_zone, cy_zone-4, size=16, color=g_color)

    # ─────────────────────────────────────────────
    # 가이드 오버레이
    # ─────────────────────────────────────────────
    def _draw_guide_overlay(self, frame, w, h, sx):
        guide   = GESTURE_GUIDE.get(self.mode, [])
        panel_w = 380
        panel_h = 40 + len(guide)*28 + 20
        px, py  = 16, TOPBAR_H + 10

        UIHelper.panel(frame, (px, py), (px+panel_w, py+panel_h), alpha=0.88, radius=12)
        UIHelper.put_text_cv(frame, f"제스처 가이드 — {self.mode} 모드",
                             (px+14, py+22), size=15, color=UI_ACCENT_AMBER)
        cv2.line(frame, (px+10, py+30), (px+panel_w-10, py+30), (70, 70, 90), 1)

        for i, (gesture, effect) in enumerate(guide):
            row_y  = py + 50 + i*28
            chip_w = 120
            UIHelper.rounded_rect(frame, (px+12, row_y-12), (px+12+chip_w, row_y+8), (50, 50, 70), 6, -1)
            UIHelper.put_text_cv(frame, gesture,       (px+18,  row_y+12), size=14, color=UI_ACCENT_CYAN)
            UIHelper.put_text_cv(frame, f"->  {effect}", (px+140, row_y+12), size=14, color=UI_TEXT_WHITE)

    # ─────────────────────────────────────────────
    # 마우스 핸들러 – 메인메뉴(jw10) + 워크스테이션(b.py) 통합
    # ─────────────────────────────────────────────
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.app_state == "MAIN_MENU":
                if "START_BTN" in self.mouse_hitboxes:
                    x1, y1, x2, y2 = self.mouse_hitboxes["START_BTN"]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.app_state = "WORKSTATION"; return
                if "EXIT_BTN" in self.mouse_hitboxes:
                    x1, y1, x2, y2 = self.mouse_hitboxes["EXIT_BTN"]
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.running = False; return
                for i in range(5):
                    box_name = f"PRESET_SLOT_{i}"
                    if box_name in self.mouse_hitboxes:
                        x1, y1, x2, y2 = self.mouse_hitboxes[box_name]
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            self.selected_preset_idx = i
                            plist = self.presets.list_presets()
                            if i < len(plist):
                                loaded = self.presets.load(plist[i])
                                with self._lock:
                                    for k, v in loaded.items():
                                        if k in self._targets: self._targets[k] = float(v)
                            return

            elif self.app_state == "WORKSTATION":
                # 워크스테이션 사이드바 버튼
                for btn_id, (bx1, by1, bx2, by2) in self.buttons_hitboxes.items():
                    if bx1 <= x <= bx2 and by1 <= y <= by2:
                        if btn_id == "EXIT":
                            self.app_state = "MAIN_MENU"
                        elif btn_id == "GUIDE":
                            self._show_guide = not self._show_guide
                        elif btn_id in ("EFFECTOR", "SYNTH"):
                            self.mode = btn_id
                        return
                # 파라미터 슬라이더 드래그 (jw10 마우스 드래그 방식)
                for name, (kx1, ky1, kx2, ky2) in self.mouse_hitboxes.items():
                    if name in self._targets and kx1 <= x <= kx2 and ky1 <= y <= ky2:
                        self.is_dragging = name
                        self._set_target_abs(name, (x-kx1)/(kx2-kx1))
                        return

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.app_state == "WORKSTATION" and self.is_dragging:
                if self.is_dragging in self.mouse_hitboxes:
                    kx1, _, kx2, _ = self.mouse_hitboxes[self.is_dragging]
                    self._set_target_abs(self.is_dragging, (x-kx1)/(kx2-kx1))

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = None

    # ─────────────────────────────────────────────
    # 손 제스처 처리 (b.py 전체 – 파라미터 조절 핵심)
    # ─────────────────────────────────────────────
    def process_gestures(self, results, frame_shape, canvas):
        h, w      = frame_shape[:2]
        ge        = self.gesture_engine
        capture_w = w - SIDEBAR_W
        left_zone, right_zone = self._get_control_zones(capture_w, h)
        left_active = right_active = False
        left_gesture = right_gesture = None

        if not results.multi_hand_landmarks:
            UIHelper.text_begin(canvas)
            self._draw_control_zone(canvas, left_zone,
                "← 왼손 제어 구역", "DELAY · REVERB · SYNTH", UI_LEFT_ZONE, False)
            self._draw_control_zone(canvas, right_zone,
                "오른손 제어 구역 →", "FILTER · DISTORTION",  UI_RIGHT_ZONE, False)
            UIHelper.text_end(canvas)
            self._hold_start.clear()
            self.precision_mode = False
            self._last_gestures = {}
            return

        lms_list        = results.multi_hand_landmarks
        handedness_list = results.multi_handedness or []
        gestures        = [ge.classify(lm.landmark) for lm in lms_list]
        self.precision_mode = any(g == "POINT" for g in gestures)

        # 양손 펼침 1초 → 초기화
        if any(g == "OPEN" for g in gestures):
            if self._check_hold("OPEN", 1.0):
                self._reset_params(); self._clear_hold("OPEN")
        else:
            self._clear_hold("OPEN")

        # 브이 0.5초 → 모드 전환
        if any(g == "PEACE" for g in gestures):
            if self._check_hold("PEACE", 0.5):
                self.mode = "SYNTH" if self.mode == "EFFECTOR" else "EFFECTOR"
                self._clear_hold("PEACE")
        else:
            self._clear_hold("PEACE")

        UIHelper.text_begin(canvas)

        for idx, (hand_lms, gesture) in enumerate(zip(lms_list, gestures)):
            lms        = hand_lms.landmark
            hand_label = "Right"
            if idx < len(handedness_list):
                hand_label = handedness_list[idx].classification[0].label

            is_left    = (hand_label == "Left")
            zone       = left_zone if is_left else right_zone
            trail      = self.trails[0 if is_left else 1]
            zone_color = UI_LEFT_ZONE if is_left else UI_RIGHT_ZONE

            itip, ttip = lms[8], lms[4]
            ix, iy = int(itip.x * w), int(itip.y * h)
            tx, ty = int(ttip.x * w), int(ttip.y * h)

            if ix > capture_w: continue

            in_zone = self._point_in_zone(ix, iy, zone)
            if is_left:
                left_active  = left_active or in_zone
                if in_zone: left_gesture = gesture
            else:
                right_active = right_active or in_zone
                if in_zone: right_gesture = gesture

            # 손 스켈레톤 (21개 랜드마크 + 연결선)
            for conn in self.mp_hands.HAND_CONNECTIONS:
                a, b = conn
                pa = (int(lms[a].x * w), int(lms[a].y * h))
                pb = (int(lms[b].x * w), int(lms[b].y * h))
                line_col = zone_color if in_zone else UI_TEXT_DIM
                cv2.line(canvas, pa, pb, tuple(c//2 for c in line_col), 1, cv2.LINE_AA)

            for lm in lms:
                cx2, cy2 = int(lm.x * w), int(lm.y * h)
                if cx2 <= capture_w:
                    cv2.circle(canvas, (cx2, cy2), 3,
                               (220, 220, 220) if in_zone else (100, 100, 100), -1, cv2.LINE_AA)

            # 양손 트레일
            trail.add((ix, iy))
            trail.draw(canvas)

            # 제스처별 시각 피드백
            g_color = GESTURE_COLORS.get(gesture, zone_color)
            if in_zone:
                if gesture == "PINCH":
                    r   = 15
                    pt1 = (ix, iy - r)
                    pt2 = (ix - int(r*0.866), iy + r//2)
                    pt3 = (ix + int(r*0.866), iy + r//2)
                    pts = np.array([pt1, pt2, pt3], np.int32)
                    cv2.drawContours(canvas, [pts], 0, g_color, -1, cv2.LINE_AA)
                    cv2.polylines(canvas, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
                elif gesture == "FIST":
                    cv2.circle(canvas, (ix, iy), 15, g_color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (ix, iy), 15, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    UIHelper.glow_circle(canvas, (ix, iy), 10, g_color, 0.6)
            else:
                cv2.circle(canvas, (ix, iy), 7, UI_TEXT_DIM, -1, cv2.LINE_AA)

            # 손목 위 제스처 레이블
            lbl_x = int(lms[0].x * w) - 30
            lbl_y = max(int(lms[0].y * h) - 28, 20)
            label = GESTURE_LABELS.get(gesture, gesture)
            if not in_zone: label = f"{label} (밖)"
            UIHelper.put_text_cv(canvas, label, (lbl_x, lbl_y), size=15,
                                 color=g_color if in_zone else UI_TEXT_DIM)

            # 핀치 연결선
            if gesture == "PINCH" and in_zone:
                cv2.line(canvas, (ix, iy), (tx, ty), (255, 255, 255), 1, cv2.LINE_AA)

            if not in_zone: continue

            norm_x, iy_norm = self._norm_in_zone(ix, iy, zone)

            # ★ 손 위치로 파라미터 조절 ★
            if is_left:
                if gesture == "PINCH":
                    self._set_target("Delay Mix",  norm_x)
                    self._set_target("Reverb Mix",  iy_norm)
                elif gesture == "FIST":
                    self._set_target("Arp BPM",    norm_x)
                    self._set_target("Synth Vol",   iy_norm)
            else:
                if gesture == "PINCH":
                    self._set_target("Filter Res",    norm_x)
                    self._set_target("Filter Cutoff", iy_norm)
                elif gesture == "FIST":
                    self._set_target("Distortion", iy_norm)
                    self._set_target("Glitch",     norm_x)

            # 제어 존 내 가이드선 + 파라미터 실시간 값 표시
            if gesture in ("PINCH", "FIST"):
                zx1, zy1, zx2, zy2 = zone
                line_color = tuple(int(c*0.7) for c in g_color)
                cv2.line(canvas, (ix, zy1), (ix, zy2), line_color, 1, cv2.LINE_AA)
                cv2.line(canvas, (zx1, iy), (zx2, iy), line_color, 1, cv2.LINE_AA)

                if is_left:
                    if gesture == "PINCH":
                        UIHelper.put_text_cv(canvas, f"딜레이 Mix: {norm_x*100:.0f}%",
                                             (zx1+10, iy-6),   size=12, color=g_color)
                        UIHelper.put_text_cv(canvas, f"리버브 Mix: {iy_norm*100:.0f}%",
                                             (ix+6,  zy1+18), size=12, color=g_color)
                    elif gesture == "FIST":
                        UIHelper.put_text_cv(canvas, f"아르페지오: {60+norm_x*180:.0f} BPM",
                                             (zx1+10, iy-6),   size=12, color=g_color)
                        UIHelper.put_text_cv(canvas, f"신스 볼륨: {iy_norm*100:.0f}%",
                                             (ix+6,  zy1+18), size=12, color=g_color)
                else:
                    if gesture == "PINCH":
                        UIHelper.put_text_cv(canvas, f"필터 공명: Q {0.1+norm_x*4.9:.2f}",
                                             (zx1+10, iy-6),   size=12, color=g_color)
                        UIHelper.put_text_cv(canvas, f"컷오프: {20*(1000**iy_norm):.0f} Hz",
                                             (ix+6,  zy1+18), size=12, color=g_color)
                    elif gesture == "FIST":
                        UIHelper.put_text_cv(canvas, f"글리치: {norm_x*100:.0f}%",
                                             (zx1+10, iy-6),   size=12, color=g_color)
                        UIHelper.put_text_cv(canvas, f"디스토션: {iy_norm*28:.1f} dB",
                                             (ix+6,  zy1+18), size=12, color=g_color)

            # 핀치로 마스터 볼륨 조절 (하단 바 영역)
            if gesture == "PINCH":
                sv_x = (capture_w)//2 - 120
                sv_y = h - BOTTOMBAR_H + 10
                sv_w = 240
                if sv_x <= ix <= sv_x+sv_w and sv_y-20 <= iy <= sv_y+20:
                    with self._lock:
                        self._targets["Master Vol"] = float(np.clip((ix-sv_x)/sv_w, 0.0, 1.0))

        self._last_gestures = {"left": left_gesture, "right": right_gesture}

        self._draw_control_zone(canvas, left_zone,
            "← 왼손 제어 구역", "DELAY · REVERB · SYNTH", UI_LEFT_ZONE, left_active, left_gesture)
        self._draw_control_zone(canvas, right_zone,
            "오른손 제어 구역 →", "FILTER · DISTORTION",  UI_RIGHT_ZONE, right_active, right_gesture)
        UIHelper.text_end(canvas)

    # ─────────────────────────────────────────────
    # 메인메뉴 드로잉 (jw10)
    # ─────────────────────────────────────────────
    def draw_main_menu(self, canvas):
        self.mouse_hitboxes.clear()
        color = (0, 255, 65)
        cv2.putText(canvas, "TOUCHLESS WORKSTATION v13", (50, 80),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
        cv2.putText(canvas, "MAIN SYSTEM MENU", (50, 120),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (180, 180, 180), 1)
        cv2.line(canvas, (50, 140), (640-50, 140), color, 1)

        # LAUNCH 버튼
        bx1, by1, bx2, by2 = 50, 180, 250, 240
        self.mouse_hitboxes["START_BTN"] = (bx1, by1, bx2, by2)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 40, 10), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(canvas, "LAUNCH ENGINE", (70, 218), cv2.FONT_HERSHEY_PLAIN, 1.1, color, 2)

        # EXIT 버튼
        ex1, ey1, ex2, ey2 = 50, 260, 250, 320
        self.mouse_hitboxes["EXIT_BTN"] = (ex1, ey1, ex2, ey2)
        cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), (10, 10, 40), -1)
        cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), (0, 100, 255), 1)
        cv2.putText(canvas, "EXIT SYSTEM", (85, 298), cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 100, 255), 1)

        # 프리셋 슬롯
        cv2.putText(canvas, "LOAD SYSTEM PRESET SLOTS:", (320, 180),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        plist = self.presets.list_presets()
        for i in range(5):
            px1, py1, px2, py2 = 320, 200+(i*45), 600, 235+(i*45)
            self.mouse_hitboxes[f"PRESET_SLOT_{i}"] = (px1, py1, px2, py2)
            is_sel  = (self.selected_preset_idx == i)
            bg_col  = (0, 70, 20)    if is_sel else (30, 30, 30)
            brd_col = (0, 255, 65)   if is_sel else (100, 100, 100)
            cv2.rectangle(canvas, (px1, py1), (px2, py2), bg_col, -1)
            cv2.rectangle(canvas, (px1, py1), (px2, py2), brd_col, 1)
            ptext = f" Slot {i+1}: {plist[i][-15:]}" if i < len(plist) else f" Slot {i+1}: [EMPTY_SLOT]"
            cv2.putText(canvas, ptext, (330, 222+(i*45)), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)

        # 단축키 안내
        hint_y = 370
        cv2.putText(canvas, "단축키 (워크스테이션 내): M=모드  N=음계  S=저장  G=가이드  1-5=불러오기  ESC=종료",
                    (50, hint_y), cv2.FONT_HERSHEY_PLAIN, 0.85, (120, 120, 150), 1)

    # ─────────────────────────────────────────────
    # 워크스테이션 UI 드로잉 (b.py 사이드바 방식)
    # ─────────────────────────────────────────────
    def draw_workstation_ui(self, frame, w, h):
        p     = self._get_smooth()
        color = self._reactive_color()
        sw    = SIDEBAR_W
        sx    = w - sw

        # 사이드바 배경
        UIHelper.panel(frame, (sx, 0), (w, h), alpha=0.92, radius=0)

        # ── 상단 타이틀 바
        cv2.rectangle(frame, (0, 0), (w, TOPBAR_H), (15, 15, 25), -1)
        cv2.line(frame, (0, TOPBAR_H), (w, TOPBAR_H), tuple(c//2 for c in color), 1)
        cv2.putText(frame, "TOUCHLESS", (18, 28), cv2.FONT_HERSHEY_DUPLEX, 0.80, UI_ACCENT_CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, " WORKSTATION v13", (120, 28), cv2.FONT_HERSHEY_DUPLEX, 0.80, UI_TEXT_WHITE, 1, cv2.LINE_AA)

        # FPS
        now = time.time()
        self._fps_buf.append(1.0 / max(now - self._prev_time, 0.001))
        self._prev_time = now
        fps       = float(np.mean(self._fps_buf))
        fps_color = UI_ACCENT_LIME if fps >= 20 else UI_ACCENT_AMBER
        cv2.putText(frame, f"FPS {fps:.0f}", (sx-90, 28), cv2.FONT_HERSHEY_PLAIN, 1.0, fps_color, 1, cv2.LINE_AA)

        UIHelper.text_begin(frame)

        # ── 모드 버튼
        btn_defs = [
            ("EFFECTOR", "이펙터 모드", "마이크 → 이펙트"),
            ("SYNTH",    "신스 모드",   "아르페지오 연주"),
        ]
        btn_y = TOPBAR_H + 14
        btn_h = 56; btn_gap = 10; btn_w = sw - 24
        for btn_id, label, sublabel in btn_defs:
            bx = sx + 12; by = btn_y
            is_active = (self.mode == btn_id)
            self.buttons_hitboxes[btn_id] = (bx, by, bx+btn_w, by+btn_h)
            if is_active:
                UIHelper.rounded_rect(frame, (bx, by), (bx+btn_w, by+btn_h),
                                      tuple(c//3 for c in color), 10, -1, 0.9)
                UIHelper.rounded_rect(frame, (bx, by), (bx+btn_w, by+btn_h), color, 10, 2)
                txt_c = UI_TEXT_WHITE
            else:
                UIHelper.rounded_rect(frame, (bx, by), (bx+btn_w, by+btn_h), (35, 35, 50), 10, -1)
                UIHelper.rounded_rect(frame, (bx, by), (bx+btn_w, by+btn_h), (60, 60, 80), 10, 1)
                txt_c = UI_TEXT_DIM
            UIHelper.put_text_cv(frame, label,    (bx+14, by+24), size=16, color=txt_c)
            UIHelper.put_text_cv(frame, sublabel, (bx+14, by+42), size=13,
                                 color=tuple(c*2//3 for c in txt_c))
            btn_y += btn_h + btn_gap

        # ── 파라미터 슬라이더 (마우스 드래그도 지원 – mouse_hitboxes 등록)
        self.mouse_hitboxes.clear()
        param_y    = btn_y + 6
        header_txt = "─ 파라미터 ─"
        UIHelper.text_center(frame, header_txt, sx + sw//2, param_y+8, size=14, color=UI_TEXT_DIM)
        param_y += 20

        for name, sp in self._smooth.items():
            if name == "Master Vol": continue
            val    = sp.value
            meta   = self._PARAM_META.get(name, (name, "■", UI_ACCENT_CYAN))
            kr_name, icon, p_color = meta
            fval   = self.format_param(name, val)

            px = sx + 10
            bar_y = param_y + 2
            bar_w = sw - 20

            # 마우스 히트박스 등록 (jw10 방식)
            self.mouse_hitboxes[name] = (px, bar_y, px+bar_w, bar_y+14)

            UIHelper.draw_bar(frame, px, bar_y, bar_w, 14, val, p_color)
            UIHelper.put_text_cv(frame, kr_name, (px, param_y+12), size=13, color=UI_TEXT_WHITE)
            tw, _ = UIHelper.text_size(fval, 13)
            UIHelper.put_text_cv(frame, fval, (sx+sw-tw-10, param_y+12), size=13, color=p_color)
            param_y += 30

        # ── 음계 & 프리셋 정보
        info_y = h - 170
        UIHelper.rounded_rect(frame, (sx+10, info_y), (w-10, info_y+55), (40, 40, 58), 8, -1)
        UIHelper.put_text_cv(frame, "음계", (sx+18, info_y+16), size=14, color=UI_TEXT_DIM)
        scale_kr = self.SCALE_KR.get(self.scale_name, self.scale_name)
        UIHelper.put_text_cv(frame, f"[N] {scale_kr}", (sx+18, info_y+36), size=15, color=UI_ACCENT_AMBER)
        plist = self.presets.list_presets()
        UIHelper.put_text_cv(frame, f"프리셋 {len(plist)}개 저장됨", (sx+18, info_y+54), size=13, color=UI_TEXT_DIM)

        # ── 가이드 토글 버튼
        gb_y = h - 108; gb_x = sx+10; gb_w = sw-20; gb_h = 30
        self.buttons_hitboxes["GUIDE"] = (gb_x, gb_y, gb_x+gb_w, gb_y+gb_h)
        g_ac = UI_ACCENT_AMBER if self._show_guide else (60, 60, 80)
        UIHelper.rounded_rect(frame, (gb_x, gb_y), (gb_x+gb_w, gb_y+gb_h),
                              g_ac if self._show_guide else (40, 40, 58), 8, -1)
        UIHelper.rounded_rect(frame, (gb_x, gb_y), (gb_x+gb_w, gb_y+gb_h), g_ac, 8, 1)
        UIHelper.text_center(frame, "도움말 닫기" if self._show_guide else "도움말 보기",
                             gb_x+gb_w//2, gb_y+16, size=14,
                             color=UI_TEXT_WHITE if self._show_guide else UI_TEXT_DIM)

        # ── EXIT 버튼 (← MENU)
        ex_y = h - 68; ex_x = sx+10; ex_w = sw-20; ex_h = 32
        self.buttons_hitboxes["EXIT"] = (ex_x, ex_y, ex_x+ex_w, ex_y+ex_h)
        UIHelper.rounded_rect(frame, (ex_x, ex_y), (ex_x+ex_w, ex_y+ex_h), (70, 25, 25), 8, -1)
        UIHelper.rounded_rect(frame, (ex_x, ex_y), (ex_x+ex_w, ex_y+ex_h), (200, 60, 60), 8, 1)
        UIHelper.text_center(frame, "< MENU (ESC)", ex_x+ex_w//2, ex_y+17, size=14, color=(220, 100, 100))

        # ── 하단 마스터 볼륨 바
        mv       = p["Master Vol"]
        bbar_y   = h - BOTTOMBAR_H
        cv2.rectangle(frame, (0, bbar_y), (sx, h), (12, 12, 20), -1)
        cv2.line(frame, (0, bbar_y), (sx, bbar_y), tuple(c//3 for c in color), 1)

        vol_x = sx//2 - 150; vol_w = 300; vol_y = bbar_y + 14
        UIHelper.put_text_cv(frame, f"마스터 볼륨  {int(mv*100)}%",
                             (vol_x-10, vol_y+12), size=15, color=UI_TEXT_WHITE)
        UIHelper.draw_bar(frame, vol_x, vol_y+2, vol_w, 16, mv, color, (35, 35, 50), radius=4)
        # 볼륨 핸들
        handle_x = vol_x + int(mv*vol_w)
        cv2.circle(frame, (handle_x, vol_y+10), 10, UI_TEXT_WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, (handle_x, vol_y+10),  8, color, -1, cv2.LINE_AA)
        # 마스터 볼륨 마우스 히트박스
        self.mouse_hitboxes["Master Vol"] = (vol_x, vol_y+2, vol_x+vol_w, vol_y+18)

        # 단축키 힌트
        UIHelper.put_text_cv(frame, "M=모드  N=음계  S=저장  G=가이드  1-5=프리셋",
                             (vol_x+vol_w+20, vol_y+14), size=12, color=UI_TEXT_DIM)

        # ── 정밀 모드 표시
        if self.precision_mode:
            UIHelper.put_text_cv(frame, "정밀 모드 (검지)", (18, bbar_y-4), size=15, color=UI_ACCENT_PINK)

        # ── 가이드 오버레이
        if self._show_guide:
            self._draw_guide_overlay(frame, w, h, sx)

        UIHelper.text_end(frame)

    # ─────────────────────────────────────────────
    # 오디오 디바이스 탐색 (b.py 개선판)
    # ─────────────────────────────────────────────
    def _find_audio_devices(self):
        devices = sd.query_devices()
        in_idx = out_idx = None
        try:
            default_in, default_out = sd.default.device[0], sd.default.device[1]
        except:
            default_in = default_out = 0

        OUT_NAMES = ["스피커", "speakers", "built-in output", "headphones",
                     "headset", "default", "realtek", "다중"]
        for i, d in enumerate(devices):
            name_l = d['name'].lower()
            if ("blackhole" in name_l or "virtual" in name_l or "cable" in name_l) and d['max_input_channels'] > 0:
                if in_idx is None: in_idx = i
            if d['max_output_channels'] > 0:
                for n in OUT_NAMES:
                    if n in name_l and out_idx is None: out_idx = i

        chosen_in  = in_idx  if in_idx  is not None else default_in
        chosen_out = out_idx if out_idx is not None else default_out
        s_rate = DEFAULT_SAMPLE_RATE
        try:
            dev_info = sd.query_devices(chosen_out)
            s_rate   = int(dev_info.get('default_samplerate', DEFAULT_SAMPLE_RATE))
        except: pass
        return chosen_in, chosen_out, s_rate

    # ─────────────────────────────────────────────
    # 메인 루프
    # ─────────────────────────────────────────────
    def run(self):
        global SIDEBAR_W, TOPBAR_H, BOTTOMBAR_H
        SIDEBAR_W   = 260
        TOPBAR_H    = 50
        BOTTOMBAR_H = 50

        in_idx, out_idx, self.samplerate = self._find_audio_devices()
        self.filter.samplerate = self.samplerate
        self.delay.samplerate  = self.samplerate
        self.synth.samplerate  = self.samplerate
        self.arp.samplerate    = self.samplerate

        win_name = "TOUCHLESS WORKSTATION"
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, self.on_mouse)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] 카메라를 열 수 없습니다."); return

        # 오디오 스트림 시작 (b.py 방식 – 입력 실패 시 출력 전용)
        stream_kwargs = dict(
            device=(in_idx, out_idx),
            samplerate=self.samplerate,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            callback=self.audio_callback,
        )
        try:
            stream = sd.Stream(**stream_kwargs)
            stream.start()
            print(f"[AUDIO_OK] {self.samplerate} Hz 가동 완료")
        except Exception as e:
            print(f"[AUDIO_WARN] 풀 스트림 실패({e}), 출력 전용 모드 시도")
            try:
                self.mode = "SYNTH"
                stream = sd.OutputStream(
                    device=out_idx, samplerate=self.samplerate,
                    blocksize=BLOCK_SIZE, channels=CHANNELS,
                    callback=self.audio_callback)
                stream.start()
                print(f"[AUDIO_OK] 출력 전용 모드 {self.samplerate} Hz")
            except Exception as e2:
                print(f"[AUDIO_ERROR] {e2}"); cap.release(); return

        self.running = True
        try:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((480, 640+SIDEBAR_W, 3), dtype=np.uint8)
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                if self.app_state == "MAIN_MENU":
                    menu_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                    self.draw_main_menu(menu_canvas)
                    cv2.imshow(win_name, menu_canvas)

                elif self.app_state == "WORKSTATION":
                    canvas  = self.render_ascii(frame)
                    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb)
                    self.process_gestures(results, frame.shape, canvas)
                    self.draw_workstation_ui(canvas, w, h)
                    cv2.imshow(win_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:                                    # ESC
                    if self.app_state == "WORKSTATION":
                        self.app_state = "MAIN_MENU"
                    else:
                        self.running = False
                elif self.app_state == "WORKSTATION":
                    if key == ord('m'):
                        self.mode = "SYNTH" if self.mode == "EFFECTOR" else "EFFECTOR"
                    elif key == ord('n'):
                        keys = list(self.SCALES.keys())
                        self.scale_name = keys[(keys.index(self.scale_name)+1) % len(keys)]
                        self.arp.scale  = self.SCALES[self.scale_name]
                    elif key == ord('s'):
                        self.presets.save(f"preset_{int(time.time())}", self._get_smooth())
                        print("[PRESET] 저장 완료")
                    elif key == ord('g'):
                        self._show_guide = not self._show_guide
                    elif ord('1') <= key <= ord('5'):
                        slot  = key - ord('1')
                        plist = self.presets.list_presets()
                        if slot < len(plist):
                            loaded = self.presets.load(plist[slot])
                            with self._lock:
                                for k, v in loaded.items():
                                    if k in self._targets: self._targets[k] = float(v)
                            print(f"[PRESET] {plist[slot]} 불러오기 완료")

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            try: stream.stop(); stream.close()
            except: pass
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    TouchlessWorkstation().run()