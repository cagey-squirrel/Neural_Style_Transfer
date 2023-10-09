from models.vgg19 import VGG19
import torch 
import numpy as np
from torch.optim import LBFGS
from losses.nst_loss import NST_loss
from utils import load_image, resize_img, get_output_dir_path_and_txt_file
import json
from nst import lbfgs_neural_style_transfer_transformation
import sys
#sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)
import os 
from time import time


def main(params):
    

    content_image_path      = params['content_image_path']
    style_image_path        = params['style_image_path']
    num_iterations          = params['num_iterations']
    learning_rate           = params['learning_rate']
    content_loss_weight     = params['content_loss_weight']
    style_loss_weight       = params['style_loss_weight']
    loss_treshold           = params['loss_treshold']
    style_loss_layer_weight = params['style_loss_layer_weight']
    output_dir_path         = params['output_dir_path'] 
    save_freq               = params['save_freq']   
    content_layer_index     = params['content_layer_index']   
    style_layer_indices     = params['style_layer_indices']     
    initial_image           = params['initial_image']                                                                                                                         


    network = VGG19(content_layer_index, style_layer_indices)

    # Loading content and style image
    content_image = load_image(content_image_path)
    style_image = load_image(style_image_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We use white (Gaussian) noise as initialization for output image
    # This is done instead of content or style image initializations so that multiple slightly different images can be generated with different initializations
    if initial_image == "random":
        output_image = torch.tensor(np.random.normal(0, 180, content_image.shape), dtype=torch.float, device=device)
    elif initial_image == "content":
        output_image = load_image(content_image_path)
    elif initial_image == "style":
        output_image = load_image(style_image_path)
        output_image = resize_img(output_image, content_image.shape)

    output_image = output_image.to(device)
    content_image = content_image.to(device)
    style_image = style_image.to(device)
    network.to(device)
    network.eval()

    output_image = torch.autograd.Variable(output_image, requires_grad=True)
    optimizer = LBFGS([output_image], lr=learning_rate, max_iter=num_iterations, line_search_fn='strong_wolfe')
    loss_function = NST_loss(content_loss_weight, style_loss_weight, style_loss_layer_weight)


    if not os.path.exists(output_dir_path):
        os.mkdir(os.path.join(os.getcwd), output_dir_path)
    

    # Makes a directory which names holds the info of initial params, makes a txt file naemd params.txt in that directory
    dir_output_path, txt_file = get_output_dir_path_and_txt_file(content_image_path, style_image_path, initial_image, content_loss_weight, style_loss_weight, output_dir_path)
    txt_file.writelines(json.dumps(params) + '\n')
    
    lbfgs_neural_style_transfer_transformation(content_image, style_image, output_image, network, optimizer, loss_function, num_iterations, loss_treshold, dir_output_path, save_freq)
    


if __name__ == "__main__":
    params_file = open('params.json', 'r')
    params = json.load(params_file)

    content_weight_params = [1e1, 1e2, 1e3]
    style_weight_params = [1e4, 1e6, 1e7]
    initializations_params = ['random', 'content', 'style']
    initializations_params = ['content']
    content_weight_params = [1e0, 1e1, 1e2]
    style_weight_params = [1e6, 1e7, 1e8]

    style_images = ['ben_giles.jpg']


    for style_img in style_images:
        params['style_image_path'] = 'style_images_examples/' + style_img
        
        main(params)