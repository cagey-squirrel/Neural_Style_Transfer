import numpy as np
from matplotlib import pyplot as plt
import os


def lbfgs_neural_style_transfer_transformation(content_image, style_image, output_image, network, optimizer, loss_function, num_iterations, loss_treshold, dir_output_path, save_freq):

    content_image_label = network(content_image, 'single')
    style_image_labels = network(style_image, 'multiple')
    iteration = [0]


    def one_epoch_iteration():

        style_image_predictions = network(output_image, 'multiple')      # We need output from multiple layers
        content_image_prediction = network(output_image, 'single')  # We only need output from a single layer

        optimizer.zero_grad()
        loss = loss_function(content_image_label, content_image_prediction, style_image_labels, style_image_predictions, iteration)
        loss.backward()

        if iteration[0] % save_freq == 0:
            index = iteration[0] // save_freq
            print(f'showing image because iteration is {iteration[0]}')
            show_img(output_image, path=dir_output_path, index=index, save=True)
        
        if iteration[0] % 100 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 1.5

        return loss

    
    optimizer.step(one_epoch_iteration)


def show_img(img, path='', index=0, save=False):
    img = img.detach().cpu().numpy()
    minval = img.min()
    maxval = img.max()
    img = (img - minval) / (maxval - minval)
    plt.imshow(img)
    plt.axis('off')

    if save:
        img_path = os.path.join(path, str(index) + '.png')
        plt.savefig(img_path)
    else:
        plt.show()