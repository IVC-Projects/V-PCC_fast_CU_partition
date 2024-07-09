# 加载 opencv 和 numpy
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.util import *
import openpyxl

edges_64x64_list, edges_32x32_list, edges_32x16_list, edges_32x8_list, edges_32x4_list, edges_16x32_list, edges_16x16_list, edges_16x8_list, edges_16x4_list, edges_8x32_list, edges_8x16_list, edges_8x8_list, edges_8x4_list, edges_4x32_list, edges_4x16_list, edges_4x8_list = [
    list() for x in range(16)]
split_64x64_list, split_32x32_list, split_32x16_list, split_32x8_list, split_32x4_list, split_16x32_list, split_16x16_list, split_16x8_list, split_16x4_list, split_8x32_list, split_8x16_list, split_8x8_list, split_8x4_list, split_4x32_list, split_4x16_list, split_4x8_list = [
    list() for y in range(16)]

geo_qp = [32, 28, 24, 20, 16]
attr_qp = [42, 37, 32, 27, 22]


def calc_laplace(sample):
    h, w = sample.shape
    l_0 = 0
    l_90 = 0
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            l_0 += abs(int(sample[i, j - 1]) + int(sample[i, j + 1]) - 2 * int(sample[i, j]))
            l_90 += abs((int(sample[i - 1, j]) + int(sample[i + 1, j]) - 2 * int(sample[i, j])))
    l_0 /= (w - 2) * (h - 2)
    l_90 /= (w - 2) * (h - 2)
    if l_90 == 0:
        l_90 = 0.0001
    return l_0 / l_90


def auto_canny(imagePath, sigma=0.33):
    gray, pic_width, pic_height = c_getYdata(imagePath)
    gray.astype(np.uint8)
    # # 进行高斯模糊去噪
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # v = np.median(blurred)
    # lower = int(max(0, (1-sigma) * v))
    # upper = int(min(255, (1+sigma) * v))
    lower = 50
    upper = 100
    auto = cv2.Canny(gray, lower, upper)
    return auto


def make_dataset(imgPath, split_file_path):
    pic_np, pic_width, pic_height = c_getYdata(imgPath)
    pic_np = pic_np.astype(np.uint8)
    edge_pic = auto_canny(imgPath)
    edge_np = np.array(edge_pic).reshape(pic_height, pic_width)
    # plt.imsave('soldier_attr_edge.png', edge_np, cmap='gray')
    edge_list = [0] * (pic_width * pic_height // 4 // 4)
    stride = pic_width // 4
    for i in range(pic_height):
        for j in range(pic_width):
            if edge_pic[i][j] == 255:
                idx = i // 4 * stride + j // 4
                edge_list[idx] += 1
    edge_np = np.array(edge_list).reshape(pic_height // 4, pic_width // 4)
    plt.figure()
    plt.imshow(edge_np[64:128,64:128], cmap='Set3')
    plt.show()
    afe
    with open(split_file_path, "r") as f:
        for line in f:
            vals = line.strip().split()
            if vals == []:
                continue
            x = int(vals[0])
            y = int(vals[1])
            width = int(vals[2])
            height = int(vals[3])
            split = vals[4]
            if split == 'CU_HORZ_SPLIT':
                split = 0
            if split == 'CU_QUAD_SPLIT':
                split = 1
            if split == 'CU_TRIH_SPLIT':
                split = 2
            if split == 'CU_TRIV_SPLIT':
                split = 3
            if split == 'CU_VERT_SPLIT':
                split = 4
            if split == 'NO_SPLIT':
                split = 5

            if x + width > pic_width or y + height > pic_height:
                continue
            if (width < 8 and height < 8) or height > 64 or width > 64:
                continue

            list_size = (width // 4) * (height // 4)
            edge_list = [0] * (list_size)
            stride = width // 4
            # 加入边缘点信息
            for i in range(height):
                for j in range(width):
                    if edge_pic[y + i][x + j] == 255:
                        idx = i // 4 * stride + j // 4
                        edge_list[idx] += 1

            # 梯度信息
            edge_list.append(calc_laplace(pic_np[y:y + height, x:x + width]))
            pic_basename = os.path.basename(split_file_path).split('.')[0]
            ctc_level_str = pic_basename.split('_')[0]
            frame_type = pic_basename.split('_')[-1]
            pic_type = pic_basename.split('_')[-2]
            ctc_level = int(ctc_level_str[1:2])
            # 几何/属性、帧类型、qp
            if pic_type == 'geo':
                edge_list.append(0)
                edge_list.append(geo_qp[ctc_level - 1])
            elif pic_type == 'attr':
                edge_list.append(1)
                edge_list.append(attr_qp[ctc_level - 1])
            # if frame_type == 'I':
            #     edge_list.append(0)
            # else:
            #     edge_list.append(1)
            if width == 64 and height == 64:
                edges_64x64_list.append(edge_list)
                split_64x64_list.append(split)
            elif width == 32 and height == 32:
                edges_32x32_list.append(edge_list)
                split_32x32_list.append(split)
            elif width == 32 and height == 16:
                edges_32x16_list.append(edge_list)
                split_32x16_list.append(split)
            elif width == 32 and height == 8:
                edges_32x8_list.append(edge_list)
                split_32x8_list.append(split)
            elif width == 32 and height == 4:
                edges_32x4_list.append(edge_list)
                split_32x4_list.append(split)
            elif width == 16 and height == 32:
                edges_16x32_list.append(edge_list)
                split_16x32_list.append(split)
            elif width == 16 and height == 16:
                edges_16x16_list.append(edge_list)
                split_16x16_list.append(split)
            elif width == 16 and height == 8:
                edges_16x8_list.append(edge_list)
                split_16x8_list.append(split)
            elif width == 16 and height == 4:
                edges_16x4_list.append(edge_list)
                split_16x4_list.append(split)
            elif width == 8 and height == 32:
                edges_8x32_list.append(edge_list)
                split_8x32_list.append(split)
            elif width == 8 and height == 16:
                edges_8x16_list.append(edge_list)
                split_8x16_list.append(split)
            elif width == 8 and height == 8:
                edges_8x8_list.append(edge_list)
                split_8x8_list.append(split)
            elif width == 8 and height == 4:
                edges_8x4_list.append(edge_list)
                split_8x4_list.append(split)
            elif width == 4 and height == 32:
                edges_4x32_list.append(edge_list)
                split_4x32_list.append(split)
            elif width == 4 and height == 16:
                edges_4x16_list.append(edge_list)
                split_4x16_list.append(split)
            elif width == 4 and height == 8:
                edges_4x8_list.append(edge_list)
                split_4x8_list.append(split)


def save_np(save_dir):
    edges_64x64_np = np.array(edges_64x64_list)
    split_64x64_np = np.array(split_64x64_list)
    edges_32x32_np = np.array(edges_32x32_list)
    split_32x32_np = np.array(split_32x32_list)
    edges_32x16_np = np.array(edges_32x16_list)
    split_32x16_np = np.array(split_32x16_list)
    edges_32x8_np = np.array(edges_32x8_list)
    split_32x8_np = np.array(split_32x8_list)
    edges_32x4_np = np.array(edges_32x4_list)
    split_32x4_np = np.array(split_32x4_list)
    edges_16x32_np = np.array(edges_16x32_list)
    split_16x32_np = np.array(split_16x32_list)
    edges_16x16_np = np.array(edges_16x16_list)
    split_16x16_np = np.array(split_16x16_list)
    edges_16x8_np = np.array(edges_16x8_list)
    split_16x8_np = np.array(split_16x8_list)
    edges_16x4_np = np.array(edges_16x4_list)
    split_16x4_np = np.array(split_16x4_list)
    edges_8x32_np = np.array(edges_8x32_list)
    split_8x32_np = np.array(split_8x32_list)
    edges_8x16_np = np.array(edges_8x16_list)
    split_8x16_np = np.array(split_8x16_list)
    edges_8x8_np = np.array(edges_8x8_list)
    split_8x8_np = np.array(split_8x8_list)
    edges_8x4_np = np.array(edges_8x4_list)
    split_8x4_np = np.array(split_8x4_list)
    edges_4x32_np = np.array(edges_4x32_list)
    split_4x32_np = np.array(split_4x32_list)
    edges_4x16_np = np.array(edges_4x16_list)
    split_4x16_np = np.array(split_4x16_list)
    edges_4x8_np = np.array(edges_4x8_list)
    split_4x8_np = np.array(split_4x8_list)

    np.save(save_dir + '\\train_X\edges_64x64_np', edges_64x64_np)
    np.save(save_dir + '\\train_Y\edges_64x64_split', split_64x64_np)
    np.save(save_dir + '\\train_X\edges_32x32_np', edges_32x32_np)
    np.save(save_dir + '\\train_Y\edges_32x32_split', split_32x32_np)
    np.save(save_dir + '\\train_X\edges_32x16_np', edges_32x16_np)
    np.save(save_dir + '\\train_Y\edges_32x16_split', split_32x16_np)
    np.save(save_dir + '\\train_X\edges_32x8_np', edges_32x8_np)
    np.save(save_dir + '\\train_Y\edges_32x8_split', split_32x8_np)
    np.save(save_dir + '\\train_X\edges_32x4_np', edges_32x4_np)
    np.save(save_dir + '\\train_Y\edges_32x4_split', split_32x4_np)
    np.save(save_dir + '\\train_X\edges_16x32_np', edges_16x32_np)
    np.save(save_dir + '\\train_Y\edges_16x32_split', split_16x32_np)
    np.save(save_dir + '\\train_X\edges_16x16_np', edges_16x16_np)
    np.save(save_dir + '\\train_Y\edges_16x16_split', split_16x16_np)
    np.save(save_dir + '\\train_X\edges_16x8_np', edges_16x8_np)
    np.save(save_dir + '\\train_Y\edges_16x8_split', split_16x8_np)
    np.save(save_dir + '\\train_X\edges_16x4_np', edges_16x4_np)
    np.save(save_dir + '\\train_Y\edges_16x4_split', split_16x4_np)
    np.save(save_dir + '\\train_X\edges_8x32_np', edges_8x32_np)
    np.save(save_dir + '\\train_Y\edges_8x32_split', split_8x32_np)
    np.save(save_dir + '\\train_X\edges_8x16_np', edges_8x16_np)
    np.save(save_dir + '\\train_Y\edges_8x16_split', split_8x16_np)
    np.save(save_dir + '\\train_X\edges_8x8_np', edges_8x8_np)
    np.save(save_dir + '\\train_Y\edges_8x8_split', split_8x8_np)
    np.save(save_dir + '\\train_X\edges_8x4_np', edges_8x4_np)
    np.save(save_dir + '\\train_Y\edges_8x4_split', split_8x4_np)
    np.save(save_dir + '\\train_X\edges_4x32_np', edges_4x32_np)
    np.save(save_dir + '\\train_Y\edges_4x32_split', split_4x32_np)
    np.save(save_dir + '\\train_X\edges_4x16_np', edges_4x16_np)
    np.save(save_dir + '\\train_Y\edges_4x16_split', split_4x16_np)
    np.save(save_dir + '\\train_X\edges_4x8_np', edges_4x8_np)
    np.save(save_dir + '\\train_Y\edges_4x8_split', split_4x8_np)


if __name__ == '__main__':
    file_dir = r'SplitDataset\label'
    yuv_dir = r'SplitDataset\yuv'
    save_dir = r'SplitDataset'
    log_list = sorted(os.listdir(file_dir))
    yuv_list = sorted(os.listdir(yuv_dir))
    for i in range(len(log_list)):
        print(log_list[i])
        if log_list[i].split('_')[1] != 'soldier':
            continue
        file_name_split = log_list[i].split('.')[0].split('_')
        yuv_name_split = yuv_list[i].split('.')[0].split('_')
        assert file_name_split[1] == yuv_name_split[1] and file_name_split[3][0:2] == yuv_name_split[4][
                                                                                      0:2], "!!!!!! log：" + \
                                                                                            log_list[
                                                                                                i] + '   yuv: ' + \
                                                                                            yuv_list[i]
        make_dataset(os.path.join(yuv_dir, yuv_list[i]), os.path.join(file_dir, log_list[i]))
    # save_np(save_dir)
