import tensorflow as tf

# モデルをコンバートする
# saved_model_dirは変換したいモデルのディレクトリ
converter = tf.lite.TFLiteConverter.from_saved_model("model.h5") 
tflite_model = converter.convert()

# 変換したモデルを保存する
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
