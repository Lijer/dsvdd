from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn import preprocessing

def dataLoading(path, logfile=None):
    file_type = path.split('.')[-1]
    print(path)
    # loading data
    df = pd.read_csv(path)
    labels = df['class']
    x_df = df.drop(['class'], axis=1)
    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)
    if logfile:
        logfile.write("Data shape: (%d, %d)\n" % x.shape)

    return x, labels

def get_KDDCup99(data_dir, batch_size):
    """Returning train and test dataloaders."""
    data_dir = data_dir
    # data = KDDCupData(data_dir, 'train')
    # dataloader_train = DataLoader(data, batch_size=args.batch_size, 
    #                           shuffle=True, num_workers=0)
    
    # test = KDDCupData(data_dir, 'test')
    # dataloader_test = DataLoader(data, batch_size=args.batch_size, 
    #                           shuffle=False, num_workers=0)
    
    features, labels = dataLoading(data_dir, None)
    features = preprocessing.scale(features)
    # features, labels = dataLoading_mat(data_dir, None)
    print(features.shape,labels.shape)
    #In this case, "atack" has been treated as normal data as is mentioned in the paper
    normal_data = features[labels==0] 
    normal_labels = labels[labels==0]

    # n_train = int(normal_data.shape[0]*0.7)
    # ixs = np.arange(normal_data.shape[0])
    # np.random.shuffle(ixs)
    # normal_data_test = normal_data[ixs[n_train:]]
    # normal_labels_test = normal_labels[ixs[n_train:]]
    normal_data_train, normal_data_test, normal_labels_train, normal_labels_test = \
        train_test_split(normal_data, normal_labels, test_size = 0.3, random_state=0, stratify = normal_labels)
    anomalous_data = features[labels==1]
    anomalous_labels = labels[labels==1]
    normal_data_test = np.concatenate((anomalous_data, normal_data_test), axis=0)
    normal_labels_test = np.concatenate((anomalous_labels, normal_labels_test), axis=0)

    test_num = normal_labels_test.shape[0]
    train_num = normal_labels_train.shape[0]

    idx_train = torch.from_numpy(np.arange(train_num))
    idx_test = torch.from_numpy(np.arange(test_num))

    normal_data_train, normal_labels_train = torch.from_numpy(np.array(normal_data_train)).float(), torch.from_numpy(np.array(normal_labels_train)).float()
    normal_data_test, normal_labels_test = torch.from_numpy(np.array(normal_data_test)).float(), torch.from_numpy(np.array(normal_labels_test)).float()
    data_train = TensorDataset(normal_data_train, normal_labels_train, idx_train)
    data_test = TensorDataset(normal_data_test, normal_labels_test, idx_test)

    dataloader_train = DataLoader(data_train, batch_size = batch_size, 
                              shuffle=True, num_workers=0)
    
    
    dataloader_test = DataLoader(data_test, batch_size = test_num, 
                              shuffle=False, num_workers = 0)
    return dataloader_train, dataloader_test

def load_dataset(dataset_name, data_path, normal_class, batch_size = 128):
    """Loads the dataset."""

    # implemented_datasets = ('mnist', 'cifar10')
    # assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name.split('.')[-1] == 'csv':
        dataset = get_KDDCup99(data_path, batch_size)

    return dataset
