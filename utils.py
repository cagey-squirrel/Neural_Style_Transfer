from PIL import Image
import numpy as np
import torch
from skimage.transform import resize
from matplotlib import pyplot as plt 
import os
from time import time


def get_output_dir_path_and_txt_file(content_image_path, style_image_path, initial_image, content_loss_weight, style_loss_weight, output_dir_path):
    '''
    Makes output dir and txt file
    '''
    
    content_image_name = content_image_path.split('/')[-1][:-4]
    style_img_name = style_image_path.split('/')[-1][:-4]
    dir_name = content_image_name + '_' + style_img_name + '_' + initial_image + '_' + str(content_loss_weight) + '_' + str(style_loss_weight) + '_' + str(time())
    dir_output_path = os.path.join(output_dir_path, dir_name)
    os.mkdir(dir_output_path)
    txt_file_path = os.path.join(dir_output_path, "params.txt")
    txt_file = open(txt_file_path, 'w')

    return dir_output_path, txt_file


def load_image(image_path, shape=512):
    '''
    Loads image and rescales it so its width has the value of shape
    This is done to ensure that both content 

    Img pixel values are put in [-256, 256] range (high values produce stronger colors, [0, 1] range produces grayish image)
    '''

    img = np.array(Image.open(image_path))

    old_width, old_height, channels = img.shape
    new_width = shape
    new_height = int(old_height * new_width / old_width)
    img = resize(img, (new_width, new_height, channels))
    img *= 512
    img = (img - 256)  # Subtracting mean
    img = np.transpose(img, (2, 0 , 1))
    img = torch.Tensor(img)

    return img


def resize_img(img, new_shape):
    '''
    Resizes tensor img into a new tensor with shape 'new_shape'
    '''

    img = img.detach().cpu().numpy()
    img = resize(img, new_shape)
    img = torch.Tensor(img)

    return img


def get_gram_matrix(layer):
    '''
    Makes gram matrix out of ouput from nn
    '''
    a, b, c, d = 1, *layer.size()  # a=batch size(=1)
    features = layer.view(a * b, c * d)  # resize F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product

    return G / (a * b * c * d)