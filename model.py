import torch
from torch import nn
import math
import torch.nn.functional as F

# Basic layers / Convolution, Activation, and Batch Normalization
def conv(ic,oc,k,s,p):
    return nn.Conv2d(ic,oc,k,s,p, bias = False)

# need to be checked
def act():
    return nn.ReLU()

def bn(ic):
    return nn.BatchNorm2d(ic, momentum = 0.1, affine = True)


# Convolution Block for Simplicity
class ConvBlock(nn.Module):
    def __init__(self,ic,oc,k,s,p):
        super(ConvBlock, self).__init__()
        self.model = nn.Sequential(
            conv(ic,oc,k,s,p),
            bn(oc)
        )
    
    def forward(self, x):
        return self.model(x)
    
# Residual Block // bottleneck & downsample supported    
class ResBlock(nn.Module):
    def __init__(self,
                in_channel,
                mid_channel,
                out_channel,
                bottleneck = False,
                downsample = False):
        super(ResBlock, self).__init__()
        
        self.downsample = downsample
        
        if downsample:
            mid_stride = 2
        else:
            mid_stride = 1
        
        if bottleneck:
            self.model_main = nn.Sequential(
                ConvBlock(in_channel, mid_channel, 1,1,0),
                act(),
                ConvBlock(mid_channel, mid_channel,3,mid_stride, 1),
                act(),
                ConvBlock(mid_channel, out_channel, 1,1,0)
                
            )
            
        else:
            self.model_main = nn.Sequential(
                ConvBlock(in_channel, mid_channel, 3, mid_stride, 1),
                act(),
                ConvBlock(mid_channel, out_channel, 3, 1, 1)
            
            )
        
        # Shortcut with 1x1 convolution // stride 2 convolution for downsampling
        if downsample:
            self.model_sc = ConvBlock(in_channel, out_channel, 1, mid_stride, 0)

    
    def forward(self, x):
        x_sc = x
        if self.downsample:
            x_sc = self.model_sc(x)
        x = self.model_main(x)
        
        return act()(x+x_sc)

    
# Residual Network
# config >> [layer_config, bottleneck]
class ResNet(nn.Module):
    def __init__(self,
                config, emb_size = 512):
        super(ResNet, self).__init__()
        self.init_channel = 64
        self.init_conv = ConvBlock(3,self.init_channel,7,1,3)
        
        self.layer, last_channel = self.get_layer(config)
        self.model = nn.Sequential(*self.layer)
        
        self.avgpool= nn.AvgPool2d(7,1)
        
        # BN -> FC -> BN
        self.last_fcn = nn.Sequential(
            nn.BatchNorm1d(last_channel, momentum = 0.1),
            nn.Dropout(0.2),
            nn.Linear(last_channel, emb_size),
            nn.BatchNorm1d(emb_size, momentum = 0.1)

        )
        
        
    # Helper function to construct each layers per stage
    def get_layer(self, config):
        layer = []
        layer_config, bottleneck = config[:-1], config[-1]
        ic = self.init_channel
        mc = ic
        oc = ic
        
        for layer_per_stage in layer_config:
            for idx in range(layer_per_stage):
                if idx==0:
                    downsample = True
                else:
                    downsample = False
                
                if bottleneck:
                    oc = mc*4
                
                layer.append(ResBlock(ic,mc,oc,bottleneck, downsample))
                ic = oc

            mc *= 2

        return layer, oc
    
    def forward(self, x):
        x = act()(self.init_conv(x))
        x = self.model(x)
        x = self.avgpool(x)
        x = nn.Flatten()(x)
        
        x = self.last_fcn(x)
        
        return x
        
        
    
# Get ResNet model with proper layer
# 18,34,50,100 supported // 152 is not implemented, but easily get by adding configs
def get_resnet(num_layer, emb_size):
    if num_layer == 18:
        config = [2,2,2,2, False]
    elif num_layer == 34:
        config = [3,4,6,3, False]
    elif num_layer == 50:
        config = [3,4,6,3, True]
    elif num_layer == 100:
        config = [3,4,23,3, True]
    else:
        raise ValueError("Not supported for %d layers, quitting..."%(num_layer))
    
    print("Consrturct ResNet with %d Layers..."%(num_layer))
    model = ResNet(config, emb_size)
    print("Done!")
    return model


if __name__ == "__main__":
    # Testing
    import torchsummary
    test_layers = [34,50]
    
    
    for num_layer in test_layers:
        model = get_resnet(num_layer)
        print("Summary for %d layers..."%(num_layer))
        torchsummary.summary(model, (3,112,112))
        