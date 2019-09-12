import torch
import torch.nn.functional as F
from torchvision import transforms
from unet import UNet
from load_data import BRATS
from utils import RandomCrop,ToTensor,DataLoader,iou,train,validate
from torch.utils.data import random_split
import os

PATH = './weights/model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=4,n_classes=4, padding=True, up_mode='upsample').to(device)
optim = torch.optim.Adam(model.parameters())

# load data
transformed_dataset = BRATS(root_dir='../BRATS/Task01_BrainTumour',
                            transform=transforms.Compose([RandomCrop((224,128)),
                                               ToTensor()]))

train_dataset,val_set = random_split(transformed_dataset,[10*len(transformed_dataset)//11,len(transformed_dataset)//11]) 
dataloader = DataLoader(train_dataset, batch_size=8,
                        shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_set, batch_size=8,
                        shuffle=True, num_workers=4)
epochs = 100

start_epoch = 0

if os.path.exists(PATH):
    state_dict = torch.load(PATH)
    model.load_state_dict(state_dict['model_state_dict'])
    optim.load_state_dict(state_dict['optimizer_state_dict'])
    start_epoch = state_dict['epoch'] + 1
    print('Loaded Weights')
    print('Resuming Training from epoch: '+ str(start_epoch))

for epoch in range(start_epoch,epochs):

    validate(model,val_dataloader,epoch,F.cross_entropy,iou,device)
    train(model,dataloader,epoch,F.cross_entropy,iou,optim,device)
    
    print("Saving Model")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }, PATH)