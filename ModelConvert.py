import tensorflow as tf

# モデルをコンバートする
# tf.lite.TFLiteConverter.from_saved_modelに入れる
# saved_model_dirは変換したいモデルのディレクトリ
converter = tf.lite.TFLiteConverter.from_saved_model("model")
tflite_model = converter.convert()  # convert

# 変換したモデルを保存する
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
