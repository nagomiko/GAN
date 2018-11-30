import scipy
import numpy as np
from tqdm import tqdm
import os


class DataLoader():
    def __init__(self, img_res):
        self.img_res = img_res
        # self.hair_type_list = []
        # self.hair_color_list = []
        # self.eye_color_list = []
        # self.img_path_list = []
        self.count = 1

        self.list_txt = []
        for i in tqdm(open('./img_list_t_not[].txt').readlines()):
            img_path, _, _, _ = i.strip().split('\t')
            if os.path.exists(img_path):
                self.list_txt.append(i.strip())
            else:
                self.count += 1

        print(self.img_res)
        print('does not load: %i' % self.count)

    def load_data(self, batch_size=32):

        batch_images = np.random.choice(self.list_txt, size=batch_size)

        img_path_list = []
        hair_type_list = []
        hair_color_list = []
        eye_color_list = []
        for i in batch_images:
            img_path, hair_type, hair_color, eye_color = i.split('\t')

            hair_type = hair_type.replace(',', '').split()
            hair_color = hair_color.replace(',', '').split()
            eye_color = eye_color.replace(',', '').split()

            hair_type_int = [int(i) for i in hair_type]
            hair_color_int = [int(i) for i in hair_color]
            eye_color_int = [int(i) for i in eye_color]

            img_path_list.append(img_path)
            hair_type_list.append(hair_type_int)
            hair_color_list.append(hair_color_int)
            eye_color_list.append(eye_color_int)

        imgs_list = []
        for img_path in img_path_list:
            img = self.imread(img_path)

            img = scipy.misc.imresize(img, self.img_res)

            imgs_list.append(img)

        imgs_list = np.array(imgs_list) / 127.5 - 1.
        hair_type_list = np.array(hair_type_list, dtype='int32')
        hair_color_list = np.array(hair_color_list, dtype='int32')
        eye_color_list = np.array(eye_color_list, dtype='int32')
        return imgs_list, hair_type_list, hair_color_list, eye_color_list

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
