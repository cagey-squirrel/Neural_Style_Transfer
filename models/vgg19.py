import torch
from torchvision.models import vgg19
import torchvision

class VGG19(torch.nn.Module):
   
    def __init__(self, content_layer_index=21, style_layers_indices=[2, 7, 12, 21, 30]):

        self.content_layer_index = content_layer_index
        self.style_layers_indices = style_layers_indices

        super().__init__()
        vgg19_network = vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)


        # Architecture of VGG 19
        # ----------------------------------------------------------------
        #    Layer (type)               Output Shape         Param #
        # ========================================================
        # 0       Conv2d-1         [-1, 64, 224, 224]           1,792
        # 1         ReLU-2         [-1, 64, 224, 224]               0
        # 2       Conv2d-3         [-1, 64, 224, 224]          36,928
        # 3         ReLU-4         [-1, 64, 224, 224]               0
        # 4    MaxPool2d-5         [-1, 64, 112, 112]               0
        # 5       Conv2d-6        [-1, 128, 112, 112]          73,856
        # 6         ReLU-7        [-1, 128, 112, 112]               0
        # 7       Conv2d-8        [-1, 128, 112, 112]         147,584
        # 8         ReLU-9        [-1, 128, 112, 112]               0
        # 9   MaxPool2d-10          [-1, 128, 56, 56]               0
        #10      Conv2d-11          [-1, 256, 56, 56]         295,168
        #11        ReLU-12          [-1, 256, 56, 56]               0
        #12      Conv2d-13          [-1, 256, 56, 56]         590,080
        #13        ReLU-14          [-1, 256, 56, 56]               0
        #14      Conv2d-15          [-1, 256, 56, 56]         590,080
        #15        ReLU-16          [-1, 256, 56, 56]               0
        #16      Conv2d-17          [-1, 256, 56, 56]         590,080
        #17        ReLU-18          [-1, 256, 56, 56]               0
        #18   MaxPool2d-19          [-1, 256, 28, 28]               0
        #19      Conv2d-20          [-1, 512, 28, 28]       1,180,160
        #20        ReLU-21          [-1, 512, 28, 28]               0
        #21      Conv2d-22          [-1, 512, 28, 28]       2,359,808
        #22        ReLU-23          [-1, 512, 28, 28]               0
        #23      Conv2d-24          [-1, 512, 28, 28]       2,359,808
        #24        ReLU-25          [-1, 512, 28, 28]               0
        #25      Conv2d-26          [-1, 512, 28, 28]       2,359,808
        #26        ReLU-27          [-1, 512, 28, 28]               0
        #27   MaxPool2d-28          [-1, 512, 14, 14]               0
        #28      Conv2d-29          [-1, 512, 14, 14]       2,359,808
        #29        ReLU-30          [-1, 512, 14, 14]               0
        # ========================================================

        # Copying weights from pretraiend network

  
        
        #self.style_layers_indices = [2, 7, 12, 21, 30]
        layer_names = ['conv11', 'conv21', 'conv31', 'conv41', 'conv51']
        num_layers = len(self.style_layers_indices)
        begin_index  = 0   # at which index does this layer begin (first layer begins at 0, next layer begins where last layer ended)
        self.layers = torch.nn.Sequential()
        for index in range(num_layers):
            layer = torch.nn.Sequential()

            layer_name = layer_names[index]
            end_index  = self.style_layers_indices[index]
            
            for layer_index in range(begin_index, end_index):  # Copy layers from begin_index to end_index
                layer.add_module(layer_name + '_' + str(layer_index), vgg19_network.features[layer_index])
            
            self.layers.add_module(layer_name, layer)
            begin_index = end_index
        

        self.content_layers = torch.nn.Sequential()
        for layer_index in range(self.content_layer_index+1):
            self.content_layers.add_module(str(layer_index), vgg19_network.features[layer_index])

        for param in self.parameters():
            param.requires_grad = False


    def forward(self, x, type='single'):

        if type == 'single':
            return self.partial_forward(x)
        elif type == 'multiple':
            return self.full_forward(x)
        else:
            raise Exception(f'Type {type} is not supported')
        


    def partial_forward(self, x):
        '''
        Returns output only for chosen level
        This is used when calculating content loss
        '''
        output = x 

        for layer in self.content_layers:
            output = layer(output)
    
        return output
    

    def full_forward(self, x):
        '''
        Returns full output from each layer in vgg19 network
        This is used when calculating style loss since we do it for each layer
        '''

        layer_outputs = []
        previous_layer_output = x

        for layer in self.layers: 
            current_layer_output = layer(previous_layer_output)
            layer_outputs.append(current_layer_output)

            previous_layer_output = current_layer_output
        
        return layer_outputs
    
    
