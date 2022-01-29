import os
import torch
from torch.utils.data import Dataset, DataLoader

# MXNet for recordio
import mxnet as mx
from mxnet import recordio

# Transforms from torchvision
import torchvision
from torchvision import transforms


class FaceImgDataset(Dataset):
    def __init__(self, img_dir, idx_path, rec_path, 
                 transform = None, target_transform = None):
        
        # To read recordio dataset, which does not supported by torch, MXnet is needed
        self.data_mxnet = recordio.MXIndexedRecordIO(os.path.join(img_dir, idx_path),
                                            os.path.join(img_dir, rec_path),
                                            'r')
        self.transform = transform
        self.target_transform = target_transform
        
        # Open property files 
        f = open(img_dir + 'property', 'r')
        self.num_classes, h,w = map(int,f.readlines()[0].rstrip().split(','))
        self.shape = (3,h,w)
        f.close()

        
    def __len__(self):
        # need to be fixed
        return 490623
    
    def __getitem__(self,idx):
        # unpack dataset into header
        header, s = recordio.unpack(self.data_mxnet.read_idx(idx))
        
        # there is an invalid image on dataset.
        try:
            image = mx.image.imdecode(s).asnumpy()
            label = int(header.label)
        except:
            return self.__getitem__(1)
        

        label = torch.tensor(label)
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return (image-0.5)/0.5, label
    
def get_dataset(img_dir, idx_path, rec_path):
    # Transform: ToTensor ([0,255] integer => [0,1] float) + Standardization + rescaling
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.RandomHorizontalFlip()])
    dataset = FaceImgDataset(img_dir, idx_path, rec_path, transform = transform)
    
    return dataset