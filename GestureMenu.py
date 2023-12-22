import cv2
import mediapipe as mp
from time import time, sleep
from utils import draw_keypoints_line
import pygame

hands = mp.solutions.hands.Hands(
    max_num_hands=2,  # 最大検出数
    min_detection_confidence=0.7,  # 検出信頼度
    min_tracking_confidence=0.7,  # 追跡信頼度
)

v_cap = cv2.VideoCapture(0)  # カメラのIDを選ぶ。映らない場合は番号を変える。


target_fps = 30
# フレームごとの待機時間を計算
clock = pygame.time.Clock()
while v_cap.isOpened():
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
        # 画像の表示
    cv2.imshow("MediaPipe Hands", img)
    if cv2.waitKey(5) & 0xFF == 27:  # ESCキーが押されたら終わる
        break
    clock.tick(target_fps)

v_cap.release()
