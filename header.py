import torch
from torch import nn
import torch.nn.functional as F
import math

class PlainHeader(nn.Module):
    def __init__(self, emb_size, num_classes, *args, **kwargs):
        super(PlainHeader, self).__init__()
        
        weight = torch.zeros(num_classes, emb_size)
        self.wt = nn.Parameter(nn.init.kaiming_normal_(weight))
        
    def forward(self, x, gt):
        return nn.CrossEntropyLoss()(F.linear(x, self.wt), gt)

class CosFaceHeader(nn.Module):
    def __init__(self, emb_size, num_classes, scale=64, margin=0.35, *args, **kwargs):
        super(CosFaceHeader, self).__init__()
        weight = torch.zeros(num_classes, emb_size)
        self.wt = nn.Parameter(nn.init.kaiming_normal_(weight))
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes
        self.emb_size = emb_size
    
    def forward(self, x, gt):
        wt_norm = F.normalize(self.wt, dim=1)
        x_norm = F.normalize(x, dim=1)
        
        cos_score = F.linear(x_norm, wt_norm)
        gt_onehot = F.one_hot(gt, num_classes = self.num_classes)
        
        margin_score = cos_score - gt_onehot * self.margin 
       
        return nn.CrossEntropyLoss()(self.scale*margin_score, gt)
        
    
class ArcFaceHeader(nn.Module):
    def __init__(self,emb_size, num_classes, scale=64, margin = 0.5, *args, **kwargs):
        super(ArcFaceHeader, self).__init__()
        weight = torch.zeros(num_classes, emb_size)
        self.wt = nn.Parameter(nn.init.kaiming_normal_(weight))
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes
        self.emb_size = emb_size
        
        self.cos_y = math.cos(margin)
        self.sin_y = math.sin(margin)
        
        self.y_siny = margin * self.sin_y
    
    def forward(self, x, gt):
       
        # Assume that gt: integer
        wt_norm = F.normalize(self.wt, dim=1)
        x_norm = F.normalize(x, dim=1)
        
        cos_score = F.linear(x_norm, wt_norm)
        gt_onehot = F.one_hot(gt, num_classes = self.num_classes)
        
        cos_x = torch.gather(cos_score, 1, gt.unsqueeze(-1)).clamp(0,1)
        sin_x = torch.sqrt(1.-torch.pow(cos_x, 2)).clamp(0,1)
        
        cos_xm = cos_x*self.cos_y - sin_x*self.sin_y
        #cos_xm_s = torch.where(cos_x>-self.cos_y, cos_xm, cos_x-self.y_siny)

        marginal_score = gt_onehot*(cos_xm-cos_x)+cos_score

        return nn.CrossEntropyLoss()(self.scale * marginal_score, gt)
   

# Not Implemented YET!! / Not Tested YET!!
class AdaCosHeader(nn.Module):
    def __init__(self, **kwargs):
        super(AdaCosHeader, self).__init__()
        pass
    
    def forward(self, x):
        pass
        
class PartialFcHeader(nn.Module):
    def __init__(self, **kwargs):
        super(PartialFcHeader, self).__init__()
        pass
    
    def forward(self, x):
        pass
    
class MagFaceHeader(nn.Module):
    def __init__(self, num_classes, emb_size, scale = 64,
                 lm = 0.35, um = 1., la = 10.,ua = 110, lg = 35,*args, **kwargs):
        super(MagFaceHeader, self).__init__()
        wt = torch.zeros(num_classes, emb_size)
        self.wt = nn.Parameter(nn.init.kaiming_normal_(wt))
        self.slope = (um-lm)/(ua-la)
        self.lg = lg
        
        self.la = la
        self.lm = lm
        self.ua = ua
        
        self.num_classes = num_classes
        self.emb_size = emb_size
        
    def margin_func(self, norm):
        # Numerical Stability
        margin = self.slope*(norm-self.la)+self.lm
        g = 1./(norm+1e-8) + norm/(self.ua**2)
        
        return margin,g
    
    def forward(self, x, gt):
        norm_x = x.norm(p=2,dim=1, keepdims= True)
        x_norm = F.normalize(x, dim=1)
        wt_norm = F.normlaize(self.wt, dim = 1)
        
        cos_score = F.linear(x_norm, wt_norm)
        gt_onehot = F.one_hot()
        margin, g = self.margin_func(norm_x)
        
        cos_m = torch.cos(margin)
        sin_m = torch.sin(margin)
        
        m_sinm = margin * sin_m
        
        cos_x = torch.gather(x, 1, gt.unsqueeze(-1)).clamp(0,1)
        sin_x = torch.sqrt(1.-torch.pow(cos_x, 2)).clamp(0,1)
        
        cos_xm = cos_x*cos_m-sin_x*sin_m
        margin_x = torch.where(cos_x>-cos_m, cos_xm, cos_x-m_sinm)
        
        marginal_score = (margin_x-cos_x) * gt_onehot + cos_score

        return nn.CrossEntropyLoss()(self.scale * marginal_score, gt) + self.lg * g

def get_header(header_type, *args, **kwargs):
    header_dict = {
        "plain":PlainHeader,
        "cosface":CosFaceHeader,
        "arcface":ArcFaceHeader,
        "adacos":AdaCosHeader,
        "partialfc":PartialFcHeader,
        "magface":MagFaceHeader
    }
    
    try:
        header = header_dict[header_type]
        
    except:
        raise ValueError("Header {} does not implemented, quitting...".format(header_type))
    
    #try:
    header = header(**kwargs)
    return header

    #except:
    #raise ValueError("Invalid config, quitting...")
    
if __name__ == "__main__":
    # Testing
    emb_size = 512
    num_classes = 10
    scale = 64
    margin = 0.5
    
    arcface = get_header('arcface', emb_size=emb_size, num_classes=num_classes, scale=scale, margin=margin)
    
    x = torch.rand((10,512))
    gt = torch.tensor([i for i in range(10)])
    y = arcface(x,gt)
    print(y)

                         
                         