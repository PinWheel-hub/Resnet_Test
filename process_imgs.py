import os, cv2
import numpy as np
from utils import remove_white_cols, split_large_img

img_dir = '/data2/chen/specsample'
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
# img_files = ['I911100423.png']
save_dir = 'test_imgs'

# 图像切为方块并落盘
def main():
    for img_num, img_file in enumerate(img_files):
        img = cv2.imread(os.path.join(img_dir, img_file))
        img = remove_white_cols(img)
        rgs_l, rgs_r = split_large_img(img=img, visual_grid=0)
        save_path_l = os.path.join(save_dir, f'{os.path.splitext(img_file)[0]}', 'l')
        save_path_r = os.path.join(save_dir, f'{os.path.splitext(img_file)[0]}', 'r')
        if not os.path.exists(save_path_l):
            os.makedirs(save_path_l)
        if not os.path.exists(save_path_r):
            os.makedirs(save_path_r)
        for i, (rg_l, rg_r) in enumerate(zip(rgs_l, rgs_r)):       
            cv2.imwrite(os.path.join(save_path_l, f'{os.path.splitext(img_file)[0]}_l_{i}.jpg'), rg_l)
            cv2.imwrite(os.path.join(save_path_r, f'{os.path.splitext(img_file)[0]}_r_{i}.jpg'), rg_r)
        print('completed: ', img_file)

if __name__ == '__main__':
    main()