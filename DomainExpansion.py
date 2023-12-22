import tensorflow as tf
from utils import preprocessing_gesture_data, draw_keypoints_line
import numpy as np
import cv2
import mediapipe as mp
import pygame

pygame.mixer.init()
pygame.mixer.music.load("assets/domain_expansion.wav")
# TFLiteモデルの読み込み
interpreter = tf.lite.Interpreter(model_path="model.tflite")
# メモリ確保。これはモデル読み込み直後に必須
interpreter.allocate_tensors()
# 学習モデルの入力層・出力層のプロパティをGet.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

hands = mp.solutions.hands.Hands(
    max_num_hands=2,  # 最大検出数
    min_detection_confidence=0.7,  # 検出信頼度
    min_tracking_confidence=0.7,  # 追跡信頼度
)


def infer_gesture(data):
    processed_data = preprocessing_gesture_data([data]).astype(np.float32)
    # indexにテンソルデータのポインタをセット
    interpreter.set_tensor(input_details[0]["index"], processed_data)
    # 推論実行
    interpreter.invoke()
    # 推論結果は、output_detailsのindexに保存されている
    output_data = interpreter.get_tensor(output_details[0]["index"])
    output_data = np.array(output_data[0])
    output_data = np.where(output_data > 0.9)
    return output_data


v_cap = cv2.VideoCapture(0)  # カメラのIDを選ぶ。映らない場合は番号を変える。
movie_cap = cv2.VideoCapture("assets/domain_expansion.mp4")
target_fps = 30
# フレームごとの待機時間を計算

domain_expansion: bool = False  # 領域展開しているか否かを判別する変数
clock = pygame.time.Clock()
while True:
    if not domain_expansion:
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
            for h_id, hand_landmarks in enumerate(
                results.multi_hand_landmarks
            ):
                for c_id, hand_class in enumerate(
                    results.multi_handedness[h_id].classification
                ):
                    positions = [0] * 21
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        lm_pos = (int(lm.x * img_w), int(lm.y * img_h))
                        positions[idx] = lm_pos
                    data[hand_class.label] = positions
        else:
            data = {}

        output_data = []
        cv2.imshow("MediaPipe Hands", img)
        if len(data.keys()) == 2:
            output_data = infer_gesture(data)
            if output_data[0] == 1:
                print("領域展開: †伏魔御厨子†")
                pygame.mixer.music.play()
                domain_expansion = True
                v_cap.release()

    elif domain_expansion:
        success, img = movie_cap.read()
        if not success:
            domain_expansion = False
            movie_cap = cv2.VideoCapture("assets/domain_expansion.mp4")
            data = {}
            v_cap = cv2.VideoCapture(0)  # カメラのIDを選ぶ。映らない場合は番号を変える。

        elif success:
            cv2.imshow("MediaPipe Hands", img)

    # FPS制御
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESCキーが押されたら終わる
        break
    clock.tick(target_fps)
