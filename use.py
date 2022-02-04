from utils import use_hwd_model_img

model_path = "hwd_model.h5"

img_path = input("image path: ")

prediction = use_hwd_model_img(img_path, img_path)

print(f"predicted: {prediction}")