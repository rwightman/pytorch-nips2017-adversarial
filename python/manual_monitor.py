import numpy as np
from PIL import Image

trained = np.load('../data/univ/monitor.npy')
delta = np.tanh(trained)
delta_pix = np.round(255.0*(delta*0.5 + 0.5)).astype(np.uint8)
delta_image = np.transpose(delta_pix[0],(1,2,0))
Image.fromarray(delta_image).show()

def show_w(w):
    delta = np.tanh(w)
    delta_pix = np.round(255.0 * (delta * 0.5 + 0.5)).astype(np.uint8)
    delta_image = np.transpose(delta_pix[0], (1, 2, 0))
    Image.fromarray(delta_image).show()

show_w(trained)

big_line_of_chinese_w = trained[:,:,33:93,:]
all_big_chinese_w = np.concatenate([big_line_of_chinese_w for _ in range(5)], axis=2)[:,:,:299,:299]

show_w(all_big_chinese_w)
np.save('../data/univ/monitor_manuala.npy', all_big_chinese_w)

all_small_chinese_w = trained[:,:,130:170,0:150]
small_chinese_full_row_w = np.concatenate([all_small_chinese_w,all_small_chinese_w], axis=3)
all_small_chinese_w = np.concatenate([small_chinese_full_row_w for _ in range(8)],axis=2)[:,:,:299,:299]

show_w(all_small_chinese_w)
np.save('../data/univ/monitor_manualb.npy', all_small_chinese_w)

