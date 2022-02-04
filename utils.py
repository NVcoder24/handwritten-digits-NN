import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

def prepare_hwd_model_img(img:str):
  img_pil = Image.open(img)
  img_resized = img_pil.resize((28, 28))
  img_gray_scale = ImageOps.grayscale(img_resized)
  return np.array([np.asarray(img_gray_scale)])

def predict_hwd_model(path:str, img:np.array):
  model = tf.keras.models.load_model(path)

  prediction = model.predict(img)

  return np.argmax(prediction)

def use_hwd_model_img(path:str, img:str):
  return predict_hwd_model(path, prepare_hwd_model_img(img))
