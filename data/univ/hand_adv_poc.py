import numpy as np
from PIL import Image

hand_img = Image.open('../data/univ/hand_adv_poc3.png').convert('RGB')
hand_img.show()

data = np.asarray(hand_img, dtype=np.uint8)
data = data.astype(np.float32)

w_matrix = np.zeros((299,299,3))

w_matrix[data < 15] = -10.0
w_matrix[data > 240] = 10.0

viz_matrix = np.tanh(w_matrix)*128 + 128

Image.fromarray(viz_matrix.astype(np.uint8)).show()

np.save('../data/univ/hand_adv_poc3', np.transpose(w_matrix,(2,0,1)))
