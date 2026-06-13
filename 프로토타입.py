import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import threading
import time
import math
from pedalboard import (
    Pedalboard, LowpassFilter, HighpassFilter, Reverb, Delay,
    Phaser, Chorus, Gain
)

# --- 글로벌 제어 변수 (목표값) ---
audio_params = {
    "lp_cutoff": 18000.0,      # 로우패스 컷오프 (높을수록 원음)
    "hp_cutoff": 20.0,         # 하이패스 컷오프 (낮을수록 원음)
    "delay_time": 0.25,
    "delay_mix": 0.0,
    "delay_feedback": 0.3,
    "reverb_room_size": 0.6,
    "reverb_wet": 0.0,
    "phaser_mix": 0.0,
    "phaser_rate": 0.5,
    "chorus_mix": 0.0,
    "gain_db": 0.0,
    "filter_kill": False,      # 양손 모으면 필터 킬 (DJ 흔한 동작)
}

current_params = dict(audio_params)
is_running = True


def get_dist_raw(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def smooth(curr, target, factor):
    return curr + (target - curr) * factor


# --- 실시간 오디오 인/아웃 스레드 ---
def audio_io_thread():
    global audio_params, current_params, is_running

    sample_rate = 44100
    buffer_size = 1024
    p = pyaudio.PyAudio()

    input_device_index = None
    output_device_index = None

    for i in range(p.get_device_count()):
        try:
            dev_info = p.get_device_info_by_index(i)
            dev_name = dev_info.get("name", "")
            if "BlackHole" in dev_name and dev_info.get("maxInputChannels", 0) > 0:
                input_device_index = i
            if dev_info.get("maxOutputChannels", 0) > 0 and "MacBook" in dev_name:
                output_device_index = i
        except Exception:
            continue

    if output_device_index is None:
        try:
            output_device_index = p.get_default_output_device_info()["index"]
        except Exception:
            pass

    if input_device_index is None:
        print("⚠️ 에러: BlackHole 장치를 찾지 못했습니다.")
        return

    # --- 이펙트 체인: 클래식 DJ 셋업 ---
    hp_filter = HighpassFilter(cutoff_frequency_hz=20.0)
    lp_filter = LowpassFilter(cutoff_frequency_hz=18000.0)
    phaser_fx = Phaser(rate_hz=0.5, depth=0.5, feedback=0.3, mix=0.0)
    delay_fx = Delay(delay_seconds=0.25, feedback=0.3, mix=0.0)
    chorus_fx = Chorus(rate_hz=1.0, depth=0.3, mix=0.0)
    reverb_fx = Reverb(room_size=0.6, wet_level=0.0, dry_level=1.0)
    gain_fx = Gain(gain_db=0.0)

    board = Pedalboard([
        hp_filter,
        lp_filter,
        phaser_fx,
        delay_fx,
        chorus_fx,
        reverb_fx,
        gain_fx,
    ])

    try:
        stream = p.open(
            format=pyaudio.paFloat32, channels=1, rate=sample_rate,
            input=True, output=True,
            input_device_index=input_device_index, output_device_index=output_device_index,
            frames_per_buffer=buffer_size
        )
    except Exception as e:
        print(f"⚠️ 오디오 스트림 개방 실패: {e}")
        return

    print("🎛️ DJ 제스처 FX 가동! 손을 움직여 필터/딜레이/리버브를 컨트롤하세요.")

    # 스무딩 속도 (낮을수록 부드럽고 천천히 변함 -> 거친 클릭 노이즈 방지)
    SMOOTH = 0.08

    while is_running:
        try:
            input_bytes = stream.read(buffer_size, exception_on_overflow=False)
            if not input_bytes:
                continue

            samples = np.frombuffer(input_bytes, dtype=np.float32).copy()

            cp = current_params
            ap = audio_params

            for key in cp:
                if key == "filter_kill":
                    continue
                cp[key] = smooth(cp[key], ap[key], SMOOTH)

            # --- 필터 킬 스위치 (양손을 모으면 순간적으로 무음/필터 킬) ---
            if ap["filter_kill"]:
                cp["lp_cutoff"] = smooth(cp["lp_cutoff"], 60.0, 0.4)
            
            # --- 적용 (안전 범위 클램핑) ---
            hp_filter.cutoff_frequency_hz = max(20.0, min(cp["hp_cutoff"], 4000.0))
            lp_filter.cutoff_frequency_hz = max(60.0, min(cp["lp_cutoff"], 18000.0))

            phaser_fx.rate_hz = max(0.05, min(cp["phaser_rate"], 4.0))
            phaser_fx.mix = max(0.0, min(cp["phaser_mix"], 0.5))  # 과하지 않게 50% 캡

            delay_fx.delay_seconds = max(0.08, min(cp["delay_time"], 0.8))
            delay_fx.mix = max(0.0, min(cp["delay_mix"], 0.6))    # 60% 캡
            delay_fx.feedback = max(0.0, min(cp["delay_feedback"], 0.65))

            chorus_fx.mix = max(0.0, min(cp["chorus_mix"], 0.4))  # 살짝만

            reverb_fx.room_size = max(0.2, min(cp["reverb_room_size"], 0.95))
            reverb_fx.wet_level = max(0.0, min(cp["reverb_wet"], 0.5))  # 50% 캡 (먹먹해지지 않게)

            gain_fx.gain_db = max(-18.0, min(cp["gain_db"], 6.0))

            effected_samples = board(samples, sample_rate)
            np.clip(effected_samples, -1.0, 1.0, out=effected_samples)

            stream.write(effected_samples.astype(np.float32).tobytes())
        except Exception:
            continue

    stream.stop_stream()
    stream.close()
    p.terminate()


audio_thread = threading.Thread(target=audio_io_thread)
audio_thread.daemon = True
audio_thread.start()

# --- 컴퓨터 비전 및 제스처 상태 머신 ---
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        black_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        left_wrist, right_wrist = None, None
        audio_params["filter_kill"] = False

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[i].classification[0].label
                wrist = hand_landmarks.landmark[0]

                tips_idx = [4, 8, 12, 16, 20]
                pip_idx = [3, 6, 10, 14, 18]
                opened_count = 0
                if hand_label == "Left" and hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                    opened_count += 1
                elif hand_label == "Right" and hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                    opened_count += 1
                for t, pp in zip(tips_idx[1:], pip_idx[1:]):
                    if hand_landmarks.landmark[t].y < hand_landmarks.landmark[pp].y:
                        opened_count += 1

                z_radius = int(get_dist_raw(hand_landmarks.landmark[9], hand_landmarks.landmark[0]) * w)
                norm_size = min(1.0, max(0.05, z_radius / 220.0))

                # ============ 왼손: 필터 디제이 노브 ============
                if hand_label == "Left":
                    left_wrist = wrist

                    # y좌표 -> 로우패스 컷오프 (위로 올리면 필터 열림, 아래로 내리면 먹먹해짐 - 클래식 DJ 필터)
                    lp_t = max(0.0, min(1.0, 1.0 - wrist.y))
                    audio_params["lp_cutoff"] = 80.0 + (lp_t ** 1.8) * 17920.0

                    # x좌표 -> 하이패스 컷오프 (오른쪽으로 갈수록 저음 깎임)
                    hp_t = max(0.0, min(1.0, wrist.x - 0.3))
                    audio_params["hp_cutoff"] = 20.0 + (hp_t ** 1.5) * 3000.0

                    # 주먹 쥐면 -> 페이저 ON (스윕 느낌)
                    if opened_count <= 1:
                        audio_params["phaser_mix"] = 0.35
                        audio_params["phaser_rate"] = 0.3 + wrist.x * 2.5
                        cv2.putText(black_canvas, "PHASER SWEEP", (int(wrist.x*w)-60, int(wrist.y*h)-30),
                                     cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 2)
                        cv2.circle(black_canvas, (int(wrist.x*w), int(wrist.y*h)), z_radius + 12, (0, 255, 255), 2)
                    else:
                        audio_params["phaser_mix"] = 0.0
                        cv2.circle(black_canvas, (int(wrist.x*w), int(wrist.y*h)), z_radius, (255, 255, 0), 1)

                    cv2.putText(black_canvas, f"L: FILTER  LP {int(audio_params['lp_cutoff'])}Hz",
                                 (int(wrist.x*w)-60, int(wrist.y*h)-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 0), 1)

                # ============ 오른손: 딜레이/리버브 디제이 ============
                elif hand_label == "Right":
                    right_wrist = wrist

                    # x좌표 -> 딜레이 타임 (1/8 ~ 1/2 비트 느낌의 범위)
                    audio_params["delay_time"] = 0.10 + wrist.x * 0.45

                    # y좌표 -> 딜레이 mix (위로 올릴수록 에코 증가, 캡 60%)
                    audio_params["delay_mix"] = max(0.0, min(0.6, (1.0 - wrist.y) * 0.7))

                    # 손 크기 -> 딜레이 피드백 (가까이 가져갈수록 에코 더 오래 남음)
                    audio_params["delay_feedback"] = 0.15 + norm_size * 0.45

                    if opened_count <= 1:
                        # 주먹 -> 리버브 워시 (공간감, 과하지 않게)
                        audio_params["reverb_room_size"] = 0.85
                        audio_params["reverb_wet"] = 0.4
                        cv2.putText(black_canvas, "REVERB WASH", (int(wrist.x*w)-60, int(wrist.y*h)-30),
                                     cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 150, 255), 2)
                        cv2.circle(black_canvas, (int(wrist.x*w), int(wrist.y*h)), z_radius + 12, (0, 150, 255), 2)
                    else:
                        audio_params["reverb_wet"] = norm_size * 0.2  # 살짝만
                        audio_params["reverb_room_size"] = 0.4 + norm_size * 0.4
                        cv2.circle(black_canvas, (int(wrist.x*w), int(wrist.y*h)), z_radius, (255, 0, 255), 1)

                    cv2.putText(black_canvas, f"R: ECHO  {audio_params['delay_time']:.2f}s",
                                 (int(wrist.x*w)-50, int(wrist.y*h)-10), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)

                for lm in hand_landmarks.landmark:
                    cv2.circle(black_canvas, (int(lm.x * w), int(lm.y * h)), 2, (90, 90, 90), -1)

            # ============ 양손 인터랙션 ============
            if left_wrist is not None and right_wrist is not None:
                hands_dist = get_dist_raw(left_wrist, right_wrist)

                # 양손을 가까이 모으면 -> 필터 킬 (DJ가 자주 쓰는 빌드업 드롭 효과)
                if hands_dist < 0.15:
                    audio_params["filter_kill"] = True
                    cv2.putText(black_canvas, "FILTER KILL!", (w//2 - 100, 30),
                                 cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 50, 50), 2)
                    cv2.line(black_canvas, (int(left_wrist.x*w), int(left_wrist.y*h)),
                             (int(right_wrist.x*w), int(right_wrist.y*h)), (255, 50, 50), 2)

                # 양손이 수평으로 나란하면 -> 코러스로 살짝 와이드하게
                y_diff = abs(left_wrist.y - right_wrist.y)
                if y_diff < 0.08:
                    audio_params["chorus_mix"] = 0.25
                    cv2.putText(black_canvas, "WIDE CHORUS", (w//2 - 100, 60),
                                 cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 255, 255), 2)
                else:
                    audio_params["chorus_mix"] = 0.0
            else:
                audio_params["chorus_mix"] = 0.0

        else:
            # 손이 없을 때 -> 원음(드라이) 상태로 천천히 복귀
            audio_params["lp_cutoff"] = 18000.0
            audio_params["hp_cutoff"] = 20.0
            audio_params["delay_mix"] = 0.0
            audio_params["reverb_wet"] = 0.0
            audio_params["phaser_mix"] = 0.0
            audio_params["chorus_mix"] = 0.0
            audio_params["gain_db"] = 0.0

        # 모니터링 화면
        cv2.putText(black_canvas, f"LP: {int(current_params['lp_cutoff'])}Hz   HP: {int(current_params['hp_cutoff'])}Hz",
                     (20, h - 70), cv2.FONT_HERSHEY_PLAIN, 1.1, (200, 200, 200), 1)
        cv2.putText(black_canvas, f"Delay mix: {int(current_params['delay_mix']*100)}%   Reverb: {int(current_params['reverb_wet']*100)}%",
                     (20, h - 45), cv2.FONT_HERSHEY_PLAIN, 1.1, (200, 200, 200), 1)
        cv2.putText(black_canvas, f"Phaser: {int(current_params['phaser_mix']*100)}%   Chorus: {int(current_params['chorus_mix']*100)}%   {'KILL!' if audio_params['filter_kill'] else ''}",
                     (20, h - 20), cv2.FONT_HERSHEY_PLAIN, 1.1, (200, 200, 200), 1)

        cv2.imshow('DJ Gesture FX', black_canvas)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

is_running = False
time.sleep(0.2)
cap.release()
cv2.destroyAllWindows()