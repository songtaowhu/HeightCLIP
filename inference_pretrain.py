from infer import InferenceHelper

# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np

infer_helper = InferenceHelper(dataset = 'HeightCLIP')

img_input_path = "./data/img/test/"
img_output_path = "./data/pred/"

infer_helper.predict_dirHeightCLIP(img_input_path, img_output_path)

