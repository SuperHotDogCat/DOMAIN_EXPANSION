import cv2
import mediapipe as mp
from time import time, sleep
import pickle
import pygame
from utils import draw_keypoints_line

# https://tama-ud.hatenablog.com/entry/2023/07/09/030155 mediapipe model maker
# https://qiita.com/Kazuhito/items/222999f134b3b27418cdを参考に作ること


hands = mp.solutions.hands.Hands(
    max_num_hands=2,  # 最大検出数
    min_detection_confidence=0.7,  # 検出信頼度
    min_tracking_confidence=0.7,  # 追跡信頼度
)

v_cap = cv2.VideoCapture(0)  # カメラのIDを選ぶ。映らない場合は番号を変える。


target_fps = 30
# フレームごとの待機時間を計算
clock = pygame.time.Clock()
all_data = []
while v_cap.isOpened():
    start_time = time()
    success, img = v_cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)  # 画像を左右反転
    img_h, img_w, _ = img.shape  # サイズ取得
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        draw_keypoints_line(
            results,
            img,
        )
        # データの追加に関する処理
        data = {}

        for h_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            for c_id, hand_class in enumerate(
                results.multi_handedness[h_id].classification
            ):
                positions = [0] * 21
                for idx, lm in enumerate(hand_landmarks.landmark):
                    lm_pos = (int(lm.x * img_w), int(lm.y * img_h))
                    positions[idx] = lm_pos
                data[hand_class.label] = positions

        # これで2つのデータが入った
        # 画像の表示
    cv2.imshow("MediaPipe Hands", img)
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESCキーが押されたら終わる
        with open("gesture_data.bin", "wb") as f:
            pickle.dump(all_data, f)
        break
    if key == ord("1"):
        if len(data.keys()) == 2:
            data["Label"] = 1
            all_data.append(data)
            print(len(all_data))
    if key == ord("0"):
        if len(data.keys()) == 2:
            data["Label"] = 0
            all_data.append(data)
            print(len(all_data))
    clock.tick(target_fps)

v_cap.release()
