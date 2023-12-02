from typing import List
import numpy as np

def preprocessing_train_data(data: List[dict]):
    output = []
    for datum in data:
        left_data = np.array(datum["Left"])
        right_data = np.array(datum["Right"])
        label = np.array(datum["Label"])

        left_data = left_data - left_data[0] #Keypoints 0からの相対ベクトルを求める
        right_data = right_data - right_data[0] #Keypoints 0からの相対ベクトルを求める
        
        left_data = left_data[1:].reshape(-1) #Keypoints 0を除外する
        left_data = left_data / np.max(np.abs(left_data)) #絶対値最大値で正規化
        right_data = right_data[1:].reshape(-1)
        right_data = right_data / np.max(np.abs(right_data)) #最大値で正規化
        label = label.reshape(-1)

        processed_data = np.concatenate([left_data, right_data, label], axis = 0)
        output.append(processed_data)
    output = np.array(output) #dim: (data_size, 81)
    processed_data, label = output[:,:-1], output[:,[-1]]
    return processed_data, label

def preprocessing_gesture_data(data: List[dict]):
    #gestureのみの処理
    output = []
    for datum in data:
        left_data = np.array(datum["Left"])
        right_data = np.array(datum["Right"])

        left_data = left_data - left_data[0] #Keypoints 0からの相対ベクトルを求める
        right_data = right_data - right_data[0] #Keypoints 0からの相対ベクトルを求める
        
        left_data = left_data[1:].reshape(-1) #Keypoints 0を除外する
        left_data = left_data / np.max(np.abs(left_data)) #絶対値最大値で正規化
        right_data = right_data[1:].reshape(-1)
        right_data = right_data / np.max(np.abs(right_data)) #最大値で正規化

        processed_data = np.concatenate([left_data, right_data], axis = 0)
        output.append(processed_data)
    output = np.array(output) #dim: (data_size, 80)
    return output