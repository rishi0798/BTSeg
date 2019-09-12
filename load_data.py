import nibabel as nib
import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class BRATS(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.train_ids =os.listdir(os.path.join(self.root_dir,'imagesTr'))

    def __len__(self):
        return len(self.train_ids)*240

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,'imagesTr',
                                self.train_ids[idx//240])
        mask_name = os.path.join(self.root_dir,'labelsTr',
                                self.train_ids[idx//240]) 
        image = nib.load(img_name).get_data()[idx%240]/4096
        mask = nib.load(mask_name).get_data()[idx%240].astype('int64')
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


class BRATS_test(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.train_ids = glob(os.path.join(self.root_dir,'imagesTs','BRATS*'))

    def __len__(self):
        return len(self.train_ids)*240

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.train_ids[idx//240])
        image = nib.load(img_name).get_data()[idx%240]/4096
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample