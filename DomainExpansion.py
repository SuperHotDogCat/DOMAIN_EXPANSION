import tensorflow as tf
from utils import preprocessing_gesture_data
import numpy as np
import pickle
import cv2
from time import time, sleep
import mediapipe as mp
# TFLiteモデルの読み込み
interpreter = tf.lite.Interpreter(model_path="model.tflite")
# メモリ確保。これはモデル読み込み直後に必須
interpreter.allocate_tensors()

# 学習モデルの入力層・出力層のプロパティをGet.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#data = pickle.load(open("gesture_data.bin","rb"))
#processed_data = preprocessing_gesture_data(data).astype(np.float32)

# indexにテンソルデータのポインタをセット
#interpreter.set_tensor(input_details[0]['index'], processed_data[[10]])

# 推論実行
#interpreter.invoke()
# 推論結果は、output_detailsのindexに保存されている
#output_data = interpreter.get_tensor(output_details[0]['index'])

# landmarkの繋がり表示用
landmark_line_ids = [ 
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # 掌
    (1, 2), (2, 3), (3, 4),         # 親指
    (5, 6), (6, 7), (7, 8),         # 人差し指
    (9, 10), (10, 11), (11, 12),    # 中指
    (13, 14), (14, 15), (15, 16),   # 薬指  
    (17, 18), (18, 19), (19, 20),   # 小指
]

hands = mp.solutions.hands.Hands(
    max_num_hands=2,                # 最大検出数
    min_detection_confidence=0.7,   # 検出信頼度
    min_tracking_confidence=0.7     # 追跡信頼度
)

v_cap = cv2.VideoCapture(0)#カメラのIDを選ぶ。映らない場合は番号を変える。

target_fps = 30
# フレームごとの待機時間を計算
frame_interval = 1.0 / target_fps

domain_expansion: bool = False #領域展開しているか否かを判別する変数

while True:
    start_time = time() #FPS制御用の変数
    if not domain_expansion:
        success, img = v_cap.read()
        start_time = time()
        success, img = v_cap.read()
        if not success:
            continue
        img = cv2.flip(img, 1)          # 画像を左右反転
        img_h, img_w, _ = img.shape     # サイズ取得
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            # 検出した手の数分繰り返し
            for h_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # landmarkの繋がりをlineで表示
                data = {}
                for line_id in landmark_line_ids:
                    # 1点目座標取得
                    lm = hand_landmarks.landmark[line_id[0]]
                    lm_pos1 = (int(lm.x * img_w), int(lm.y * img_h))
                    # 2点目座標取得
                    lm = hand_landmarks.landmark[line_id[1]]
                    lm_pos2 = (int(lm.x * img_w), int(lm.y * img_h))
                # line描画
                    cv2.line(img, lm_pos1, lm_pos2, (128, 0, 0), 1)
                        # landmarkをcircleで表示
                    z_list = [lm.z for lm in hand_landmarks.landmark]
                    z_min = min(z_list)
                    z_max = max(z_list)

                    for idx, lm in enumerate(hand_landmarks.landmark):
                        lm_pos = (int(lm.x * img_w), int(lm.y * img_h))
                        lm_z = int((lm.z - z_min) / (z_max - z_min) * 255)
                        cv2.circle(img, lm_pos, 3, (255, lm_z, lm_z), -1)

                    # 検出情報をテキスト出力
                    # - テキスト情報を作成
                    hand_texts = []
                    for c_id, hand_class in enumerate(results.multi_handedness[h_id].classification):
                        hand_texts.append("#%d-%d" % (h_id, c_id)) 
                        hand_texts.append("- Index:%d" % (hand_class.index))
                        hand_texts.append("- Label:%s" % (hand_class.label))
                        hand_texts.append("- Score:%3.2f" % (hand_class.score * 100))
                # - テキスト表示に必要な座標など準備
                        lm = hand_landmarks.landmark[0] #手の甲
                        lm_x = int(lm.x * img_w) - 50
                        lm_y = int(lm.y * img_h) - 10
                        lm_c = (64, 0, 0)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        # - テキスト出力
                        for cnt, text in enumerate(hand_texts):
                            cv2.putText(img, text, (lm_x, lm_y + 10 * cnt), font, 0.3, lm_c, 1)
    #データの追加に関する処理
            data = {}
            for h_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for c_id, hand_class in enumerate(results.multi_handedness[h_id].classification):
                    positions = [0] * 21
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        lm_pos = (int(lm.x * img_w), int(lm.y * img_h))
                        positions[idx] = lm_pos
                    data[hand_class.label] = positions
        else:
            data = {}
        cv2.imshow("MediaPipe Hands", img)
        end_time = time()
        elapsed_time = end_time - start_time
        sleep_time = max(0, frame_interval - elapsed_time)
        sleep(sleep_time)
        key = cv2.waitKey(5) & 0xFF
        if key == 27:#ESCキーが押されたら終わる
            break
        if len(data.keys()) == 2:
            processed_data = preprocessing_gesture_data([data]).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], processed_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = np.array(output_data[0])
            output_data = np.where(output_data > 0.9)
            if output_data[0] == 1:
                print("領域展開: †伏魔御厨子†")
            else:
                print("領域展開できない")