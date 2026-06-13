import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import threading
import time
from pedalboard import Pedalboard, Distortion, Chorus, LowpassFilter

# --- 글로벌 제어 변수 (비전 -> 오디오 전달) ---
audio_params = {
    "frequency": 220.0,
    "volume": 0.1,
    "unison_count": 1,        # 교차점에 따라 늘어날 유니즌 보이스 수 (1~7)
    "unison_detune": 0.02,     # 미세 디튜닝 값
    "distortion_drive": 0.0,
    "eq_cutoff": 2000.0
}
is_running = True

# --- 두 선분이 교차하는지 검사하는 기하학 함수 (CCW 알고리즘) ---
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def check_intersection(p1, q1, p2, q2):
    # 두 선분 (p1-q1)과 (p2-q2)가 교차하는지 판별
    return ccw(p1, p2, q2) != ccw(q1, p2, q2) and ccw(p1, q1, p2) != ccw(p1, q1, q2)

# --- 오디오 생성 스레드 (Sine Wave + Unison + Effects) ---
def audio_stream_thread():
    global audio_params, is_running
    
    sample_rate = 44100
    buffer_size = 512
    p = pyaudio.PyAudio()
    
    board = Pedalboard()
    dist = Distortion(drive_db=0)
    chorus = Chorus(rate_hz=0.5, depth=0.2, centre_delay_ms=7.0, feedback=0.1, mix=0.0)
    lowpass = LowpassFilter(cutoff_frequency_hz=2000)
    
    board.append(dist)
    board.append(chorus)
    board.append(lowpass)
    
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True, frames_per_buffer=buffer_size)
    
    phases = np.zeros(7) # 최대 7개 유니즌 보이스의 위상 관리
    
    while is_running:
        freq = audio_params["frequency"]
        vol = audio_params["volume"]
        unison = max(1, min(int(audio_params["unison_count"]), 7))
        detune = audio_params["unison_detune"]
        
        # 이펙터 파라미터 업데이트
        dist.drive_db = audio_params["distortion_drive"]
        lowpass.cutoff_frequency_hz = max(50.0, min(audio_params["eq_cutoff"], 18000.0))
        chorus.mix = min(1.0, unison * 0.12) # 유니즌이 많아질수록 코러스(플랜저) 효과 증폭
        
        # 사인파 멀티 오실레이터 합성 (Unison 기능)
        t = np.arange(buffer_size) / sample_rate
        combined_signal = np.zeros(buffer_size)
        
        # 중심 주파수를 기준으로 미세하게 음정을 비틀어 겹침 (두텁고 거대한 소리 유도)
        for i in range(unison):
            if i == 0:
                f_offset = 0
            else:
                # i가 커질수록 양옆으로 주파수가 벌어짐
                f_offset = freq * detune * (((i + 1) // 2) * (-1 if i % 2 == 0 else 1))
                
            current_freq = max(20.0, freq + f_offset)
            
            # 각 보이스별 독립적 위상 계산으로 글리치 방지
            sig = np.sin(2 * np.pi * current_freq * t + phases[i])
            phases[i] += 2 * np.pi * current_freq * (buffer_size / sample_rate)
            phases[i] %= 2 * np.pi
            
            combined_signal += sig
            
        # 보이스 수만큼 나눠서 클리핑(찢어짐) 방지 후 볼륨 적용
        combined_signal = (combined_signal / unison) * vol
        samples = combined_signal.astype(np.float32)
        
        # Pedalboard 이펙트 통과
        effected_samples = board(samples, sample_rate)
        stream.write(effected_samples.tobytes())
        
    stream.stop_stream()
    stream.close()
    p.terminate()

# 오디오 스레드 시작
audio_thread = threading.Thread(target=audio_stream_thread)
audio_thread.daemon = True
audio_thread.start()

# --- 컴퓨터 비전 및 기하학적 트래킹 (MediaPipe) ---
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

# 양손 추적을 위해 max_num_hands=2 설정
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue
            
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        
        # [핵심] 카메라 영상을 쓰지 않고 완전한 검은색 바탕 화면을 새로 생성
        black_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # 양손의 손가락 끝 좌표들을 담을 리스트
        hand_tips_left = []
        hand_tips_right = []
        
        all_landmarks_points = []
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 왼손/오른손 라벨 확인
                hand_label = results.multi_handedness[i].classification[0].label
                
                # 모든 관절 위치를 검은 배경에 점(Dot)으로 표현
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    all_landmarks_points.append((cx, cy))
                    # 네온 블루 컬러의 점들로 관절 시각화
                    cv2.circle(black_canvas, (cx, cy), 4, (255, 255, 100), -1)
                
                # 5개 손가락 끝마디(Tip) 추출
                tips_idx = [4, 8, 12, 16, 20]
                tips_pts = []
                for idx in tips_idx:
                    lm = hand_landmarks.landmark[idx]
                    tips_pts.append((int(lm.x * w), int(lm.y * h)))
                    
                if hand_label == "Left":
                    hand_tips_left = tips_pts
                else:
                    hand_tips_right = tips_pts

            # --- 양손 손가락 연결선 생성 및 교차 검사 ---
            intersections_count = 0
            lines_left = []
            lines_right = []
            
            # 왼손 안에서의 손가락 끝끼리 연결선 조합 (예: 엄지-검지, 검지-중지 등)
            if len(hand_tips_left) >= 2:
                for idx in range(len(hand_tips_left)-1):
                    lines_left.append((hand_tips_left[idx], hand_tips_left[idx+1]))
            # 오른손 안에서의 손가락 끝끼리 연결선 조합
                for idx in range(len(hand_tips_right)-1):
                    lines_right.append((hand_tips_right[idx], hand_tips_right[idx+1]))
            
            # 양손 사이를 가로지르는 교차 연결선 (왼손 끝 i번과 오른손 끝 j번을 크로스로 연결)
            cross_lines = []
            if hand_tips_left and hand_tips_right:
                for p_l in hand_tips_left:
                    for p_r in hand_tips_right:
                        cross_lines.append((p_l, p_r))
            
            # 모든 연결선을 검은 화면에 사이버네틱한 선으로 그리기
            for line in cross_lines:
                cv2.line(black_canvas, line[0], line[1], (150, 50, 255), 1) # 자줏빛 실선
                
            # 실시간 선 교차(Crossover) 여부 수학적 계산
            if len(cross_lines) >= 2:
                for idx1 in range(len(cross_lines)):
                    for idx2 in range(idx1 + 1, len(cross_lines)):
                        l1 = cross_lines[idx1]
                        l2 = cross_lines[idx2]
                        # 두 선분이 교차하는지 검증
                        if check_intersection(l1[0], l1[1], l2[0], l2[1]):
                            intersections_count += 1
                            # 교차하는 지점을 계산해 붉은색 원으로 스파크 효과 시각화
                            mx = int((l1[0][0] + l1[1][0] + l2[0][0] + l2[1][0]) / 4)
                            my = int((l1[0][1] + l1[1][1] + l2[0][1] + l2[1][1]) / 4)
                            cv2.circle(black_canvas, (mx, my), 6, (0, 0, 255), -1)

            # --- 교차점 결과에 따른 소리 변조 매핑 (사운드 디자인) ---
            if hand_tips_left and hand_tips_right:
                # 양손의 거리에 따라 기본 주파수 결정 (두 손목 대용 엄지 간의 거리)
                dist_hands = np.linalg.norm(np.array(hand_tips_left[0]) - np.array(hand_tips_right[0]))
                audio_params["frequency"] = max(80.0, min(120.0 + dist_hands * 1.5, 1500.0))
                audio_params["volume"] = 0.15
            else:
                audio_params["volume"] = 0.01 # 손이 하나만 있거나 없으면 소리를 낮춤
                
            # 교차점이 많아질수록 소리가 무시무시해집니다.
            # 1. 유니즌(Unison) 보이스 증가: 소리가 엄청 거대하고 두터워짐
            audio_params["unison_count"] = 1 + (intersections_count // 5) 
            # 2. 디스토션 강도 증가
            audio_params["distortion_drive"] = min(25.0, intersections_count * 0.4)
            # 3. 로우패스 필터 개방 (소리가 점점 밝고 날카로워짐)
            audio_params["eq_cutoff"] = 800.0 + (intersections_count * 150.0)

            # 디스플레이 정보 출력
            cv2.putText(black_canvas, f"Crossover Links: {intersections_count}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 255), 2)
            cv2.putText(black_canvas, f"Unison Voices: {int(audio_params['unison_count'])}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 2)
            cv2.putText(black_canvas, f"Waveform: SINEWAVE", (20, 100), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 100, 100), 2)
        else:
            audio_params["volume"] = 0.0
            cv2.putText(black_canvas, "BRING BOTH HANDS IN FRONT OF CAMERA", (40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2)

        # 화면 출력 (실제 비디오는 사라지고 검은 캔버스만 노출)
        cv2.imshow('Black Canvas Hand Audio Matrix', black_canvas)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

is_running = False
time.sleep(0.2)
cap.release()
cv2.destroyAllWindows()