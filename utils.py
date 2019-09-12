import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn.functional import one_hot
import numpy as np
from tqdm import tqdm

SMOOTH = 1e-6

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        mask = mask[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}

class RandomCrop_test(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image}


class ToTensor_test(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image)}


def iou(pred,tar):
    """Calculate IOU."""

    y_pred = pred.detach()
    y_pred = (y_pred > 0.5).byte()

    y_tar = tar.detach()
    y_one_hot = one_hot(y_tar,4).permute(0,3,1,2).byte()

    intersection = (y_pred & y_one_hot).float().sum((2, 3))  # Will be zero if Truth=0 or Prediction=0
    union = (y_pred | y_one_hot).float().sum((2, 3))         # Will be zzero if both are 0
    
    iou_val = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou_val - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    thresholded = thresholded.mean(1).mean(0)

    return thresholded


def train(model, train_loader, epoch, loss_function, 
          additional_metric, optimiser, device):

    model.train()
    loop = tqdm(train_loader)
    
    for data in loop:
        X = data['image']
        target = data['mask']
        X = X.to(device)  # [N, 1, H, W]
        target = target.to(device)  # [N, H, W] with class indices (0, 4)
        
        optimiser.zero_grad()
        prediction = model(X)  # [N, 4, H, W]
        
        loss = loss_function(prediction, target)
        metric = additional_metric(prediction, target)
        
        loss.backward()
        optimiser.step()
        
        loop.set_description('Epoch {}'.format(epoch))
        loop.set_postfix(loss=loss.item(), iou=metric.item(),max = target.max())

def test(model, test_loader, device):

    model.eval()
    loop = test_loader
    
    for i,data in enumerate(loop):
        X = data['image']
        X = X.to(device)  # [N, 1, H, W]
        
        prediction = model(X)  # [N, 4, H, W]
        
        if i==0:
            break
        
    return prediction, X, one_hot(data['mask'],4).permute(0,3,1,2)