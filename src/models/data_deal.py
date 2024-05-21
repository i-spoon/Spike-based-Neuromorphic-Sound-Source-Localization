import torch
import numpy as np
import hdf5storage
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os,json
from tqdm import tqdm
class MyDataloaderClass(Dataset):
    def __init__(self, X_data, Y_data):
        self.x_data = X_data
        self.y_data = Y_data
        self.len = X_data.shape[0]

    def __getitem__(self, index):
        self.x_data = self.x_data.view(-1,4,6,51,40)
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    
def get_train_valid_loader(data_dir,random_seed=1234,batch_size=128,augment=False,valid_size=0.1,shuffle=True,num_workers=4,pin_memory=False):
    
    data=hdf5storage.loadmat(data_dir+'Training_1.5m_40channel_new_{}.mat'.format(1))
    Xtr = data['X']
    Ytr = data['Y']
    Ztr = data['Z']
    for i in range(2,11):
        data=hdf5storage.loadmat(data_dir+'Training_1.5m_40channel_new_{}.mat'.format(i))
        Xtr = np.vstack((Xtr,data['X']))
        Ytr = np.vstack((Ytr,data['Y']))
        Ztr = np.vstack((Ztr,data['Z']))
    # Zfin = []
    # for res in Ztr:
    #     Zfin.append(res)
    #     Zfin.append(res)

    Xmean = Xtr.mean(axis=1)
    # Xtr = torch.tensor(Xtr)
    Xtr = torch.tensor((Xtr.T - Xmean.reshape(1, -1)).T)
    Ytr = torch.tensor(Ytr)  # Binary target
    Ztr = torch.tensor(Ztr)  # Gaussian-format target
    print("train_shape:{}".format(Xtr.shape))
    num_train = Xtr.shape[0]
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_x = Xtr[train_idx]
    valid_x = Xtr[valid_idx]
    train_z = Ztr[train_idx]
    valid_z = Ztr[valid_idx]

    train_loader_obj = MyDataloaderClass(train_x, train_z)
    val_loader_obj = MyDataloaderClass(valid_x, valid_z)

    train_loader = DataLoader(
        dataset=train_loader_obj, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,drop_last=True
    )

    valid_loader = DataLoader(
        dataset=val_loader_obj, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,drop_last=True
    )

    return (train_loader, valid_loader)

def get_test_loader(data_dir,batch_size=128,shuffle=False,num_workers=4,pin_memory=False):
    
    data=hdf5storage.loadmat(data_dir+'Testing_1.5m_40channel_new_{}.mat'.format(1))
    Xte = data['X']
    Yte = data['Y'] 
    Zte = data['Z']
    for i in range(2,11):
        data=hdf5storage.loadmat(data_dir+'Testing_1.5m_40channel_new_{}.mat'.format(i))
        Xte = np.vstack((Xte,data['X']))
        Yte = np.vstack((Yte,data['Y']))
        Zte = np.vstack((Zte,data['Z']))
    
    # Zfin = []
    # for res in Zte:
    #     Zfin.append(res)
    #     Zfin.append(res)
    Xmean = Xte.mean(axis=1)
    # Xte = torch.tensor(Xte)
    Xte = torch.tensor((Xte.T - Xmean.reshape(1, -1)).T)
    Yte = torch.tensor(Yte)  # Binary target
    Zte = torch.tensor(Zte)  # Gaussian-format target

    print('test_shape:{}'.format(Xte.shape))

    test_loader_obj = MyDataloaderClass(Xte, Zte)
    test_loader = DataLoader(dataset=test_loader_obj, batch_size=batch_size,
                             shuffle=False, num_workers=4,drop_last=True)

    return test_loader

print('==> Preparing data..')

train_data_dir = r'/root/yxy/ws/TCJA-New/dataset/new_dataset/' 
test_data_dir =r'/root/yxy/ws/TCJA-New/dataset/new_dataset/'

# (train_loader, val_loader) = get_train_valid_loader(train_data_dir)
test_loader = get_test_loader(test_data_dir)