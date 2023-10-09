import torch 
from utils import get_gram_matrix
from nst import show_img
from matplotlib import pyplot as plt
import numpy as np 


class NST_loss(torch.nn.Module):


    def __init__(self, content_loss_weight, style_loss_weight, style_loss_layer_weight):
        super().__init__()

        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        #self.mse_loss = self.custom_mse
        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight
        self.style_loss_layer_weight = style_loss_layer_weight
    

    def custom_mse(self, label, prediction):
        
        num_values = 1
        for shape_value in label.shape:
            num_values *= shape_value

        error = (label - prediction)
        error = torch.pow(error, 4)
        return error.sum() / num_values


    def forward(self, content_image_label, content_image_prediction, style_image_labels, style_image_predictions, iteration):

        content_loss = self.mse_loss(content_image_label, content_image_prediction)

        style_loss = 0
        for style_image_label, style_image_prediction in zip(style_image_labels, style_image_predictions):
            gram_matrix_label = get_gram_matrix(style_image_label)
            gram_matrix_prediction = get_gram_matrix(style_image_prediction)
            style_layer_loss = self.mse_loss(gram_matrix_label, gram_matrix_prediction)
            style_loss += self.style_loss_layer_weight * style_layer_loss

        total_loss = self.content_loss_weight * content_loss + self.style_loss_weight * style_loss
        print(f'iteration:{iteration[0]} style_loss = {style_loss}')
        print(f'iteration:{iteration[0]} content loss = {self.content_loss_weight * content_loss} style_loss = {self.style_loss_weight * style_loss} \n')
        iteration[0] += 1

        return total_loss
