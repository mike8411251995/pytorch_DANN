import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import params
from sklearn.manifold import TSNE


import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import os, time
from data import SynDig

# From dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np
import scipy.io
import gzip
import wget
import h5py
import pickle
import urllib
import os
import skimage
import skimage.transform
from skimage.io import imread
import matplotlib.image as mpimg
import random
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import math

def get_train_loader(name, root, scale=28, shuffle=True, style=None, attr=None):
    if name == 'mnist':
        return LoadMNIST(root+'mnist/', batch_size=params.batch_size, split='train', shuffle=shuffle, scale=scale)
    elif name == 'usps':
        return LoadUSPS(root+'usps/', batch_size=params.batch_size, split='train', shuffle=shuffle, scale=scale)
    elif name == 'svhn':
        return LoadSVHN(root+'svhn/', batch_size=params.batch_size, split='extra', shuffle=shuffle, scale=scale)
    elif name == 'mnistm':
        return LoadMNISTM(root+'mnistm/', batch_size=params.batch_size, split='train', shuffle=shuffle, scale=scale)

def get_test_loader(name, root, scale=28, shuffle=True, style=None, attr=None):
    if name == 'mnist':
        return LoadMNIST(root+'mnist/', batch_size=params.batch_size, split='test', shuffle=False, scale=scale)
    elif name == 'usps':
        return LoadUSPS(root+'usps/', batch_size=params.batch_size, split='test', shuffle=False, scale=scale)
    elif name == 'svhn':
        return LoadSVHN(root+'svhn/', batch_size=params.batch_size, split='test', shuffle=False, scale=scale)
    elif name == 'mnistm':
        return LoadMNISTM(root+'mnistm/', batch_size=params.batch_size, split='test', shuffle=False, scale=scale)

def LoadSVHN(data_root, batch_size, split='train', shuffle=True, scale=28):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    trans = transforms.Compose([transforms.Resize(size=[scale, scale]),transforms.ToTensor()])

    svhn_dataset = datasets.SVHN(data_root, split=split, download=True,
                                   transform=trans)
    return DataLoader(svhn_dataset,batch_size=batch_size, shuffle=shuffle, drop_last=True)

def LoadUSPS(data_root, batch_size, split='train', shuffle=True, scale=28):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    usps_dataset = USPS_2(root=data_root,train=(split=='train'),download=True,scale=scale)
    return DataLoader(usps_dataset,batch_size=batch_size, shuffle=shuffle, drop_last=True)

def LoadMNIST(data_root, batch_size, split='train', shuffle=True, scale=28):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    trans = transforms.Compose([transforms.Resize(size=[scale, scale]),transforms.ToTensor()])
    
    mnist_dataset = datasets.MNIST(data_root, train=(split=='train'), download=True,
                                   transform=trans)
    return DataLoader(mnist_dataset,batch_size=batch_size,shuffle=shuffle, drop_last=True)

def LoadMNISTM(data_root, batch_size, split='train', shuffle=True, scale=28):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    trans = transforms.Compose([transforms.Resize(size=[scale, scale]),transforms.ToTensor()])

    mnist_dataset = MNISTM(data_root, train=(split=='train'), download=True,
                                   transform=trans)
    return DataLoader(mnist_dataset,batch_size=batch_size,shuffle=shuffle, drop_last=True)

### USPS Reference : https://github.com/corenel/torchzoo/blob/master/torchzoo/datasets/usps.py
class USPS(Dataset):
    """USPS Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://github.com/mingyuliutw/CoGAN/blob/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"
    
    def __init__(self, root, train=True, scale=28, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)

        if scale == 32:
            self.filename = "usps_32x32.pkl"
        else:
            self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]

        #self.train_data *= 255.0
        #self.train_data = self.train_data.transpose(
        #    (0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return torch.FloatTensor(img), label[0]

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, 'usps_28x28.pkl')
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if not os.path.isfile(filename):
            print("Download %s to %s" % (self.url, os.path.abspath(filename)))
            #urllib.request.urlretrieve(self.url, filename)
            wget.download(self.url,out=os.path.join(self.root, 'usps_28x28.pkl'))
            print("[DONE]")
        if not os.path.isfile(os.path.join(self.root, 'usps_32x32.pkl')):
            print("Resizing USPS 28x28 to 32x32...")
            f = gzip.open(os.path.join(self.root, 'usps_28x28.pkl.gz'), "rb")
            data_set = pickle.load(f, encoding="bytes")
            for d in [0,1]:
                tmp = []
                for img in range(data_set[d][0].shape[0]):
                    tmp.append(np.expand_dims(skimage.transform.resize(data_set[d][0][img].squeeze(),[32,32]),0))
                data_set[d][0] = np.array(tmp)
            fp=gzip.open(os.path.join(self.root, 'usps_32x32.pkl'),'wb')
            pickle.dump(data_set,fp)
            print("[DONE")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels

class DatasetParams(object):
    "Class variables defined."
    num_channels = 1
    image_size   = 16
    mean         = 0.1307
    std          = 0.3081
    num_cls      = 10
    target_transform = None

class USPSParams(DatasetParams):
    
    num_channels = 1
    image_size   = 16
    #mean = 0.1307
    #std = 0.30
    #mean         = 0.254
    #std          = 0.369
    mean = 0.5
    std = 0.5
    num_cls      = 10
    
class USPS_2(Dataset):

    """USPS handwritten digits.
    Homepage: http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html
    Images are 16x16 grayscale images in the range [0, 1].
    """

    base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'

    

    data_files = {
        'train': 'zip.train.gz',
        'test': 'zip.test.gz'
        }

    params = USPSParams()

    def __init__(self, root, train=True, target_transform=None,
            download=False,scale=28):

        transform = []
        

        transform += [transforms.Resize(size=[scale, scale]),transforms.ToTensor()]

        self.root = root
        self.train = train
        self.transform = transforms.Compose(transform)
        self.target_transform = target_transform
	
        if download:
            self.download()

        if self.train:
            datapath = os.path.join(self.root, self.data_files['train'])
        else:
            datapath = os.path.join(self.root, self.data_files['test'])

        self.images, self.targets = self.read_data(datapath)
    
    def get_path(self, filename):
        return os.path.join(self.root, filename)

    def download(self):
        data_dir = self.root
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in self.data_files.values():
            path = self.get_path(filename)
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                util.maybe_download(url, path)

    def read_data(self, path):
        images = []
        targets = []
        with gzip.GzipFile(path, 'r') as f:
            for line in f:
                split = line.strip().split()
                label = int(float(split[0]))
                pixels = np.array([(float(x) + 1) / 2 for x in split[1:]]) * 255
                num_pix = self.params.image_size
                pixels = pixels.reshape(num_pix, num_pix).astype('uint8')
                img = Image.fromarray(pixels, mode='L')
                images.append(img)
                targets.append(label)
        return images, targets

    def __getitem__(self, index):
        img = self.images[index]
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)
        
class MNISTM(Dataset):
    """`MNIST-M Dataset."""

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'mnist_m_train.pt'
    test_file = 'mnist_m_test.pt'

    def __init__(self,
                 root, mnist_root="data",
                 train=True,
                 transform=None, target_transform=None,
                 download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__()
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = \
                torch.load(os.path.join(self.root,
                                        self.processed_folder,
                                        self.training_file))
        else:
            self.test_data, self.test_labels = \
                torch.load(os.path.join(self.root,
                                        self.processed_folder,
                                        self.test_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           self.processed_folder,
                                           self.training_file)) and \
            os.path.exists(os.path.join(self.root,
                                        self.processed_folder,
                                        self.test_file))

    def download(self):
        """Download the MNIST data."""
        # import essential packages
        from six.moves import urllib
        import gzip
        import pickle
        from torchvision import datasets

        # check if dataset already exists
        if self._check_exists():
            return

        # make data dirs
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # download pkl files
        print('Downloading ' + self.url)
        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path.replace('.gz', '')):
            data = urllib.request.urlopen(self.url)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        # load MNIST-M images from pkl file
        with open(file_path.replace('.gz', ''), "rb") as f:
            mnist_m_data = pickle.load(f, encoding='bytes')
        mnist_m_train_data = torch.ByteTensor(mnist_m_data[b'train'])
        mnist_m_test_data = torch.ByteTensor(mnist_m_data[b'test'])

        # get MNIST labels
        mnist_train_labels = datasets.MNIST(root=self.mnist_root,
                                            train=True,
                                            download=True).train_labels
        mnist_test_labels = datasets.MNIST(root=self.mnist_root,
                                           train=False,
                                           download=True).test_labels

        # save MNIST-M dataset
        training_set = (mnist_m_train_data, mnist_train_labels)
        test_set = (mnist_m_test_data, mnist_test_labels)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer

def displayImages(dataloader, length=8, imgName=None):
    """
    Randomly sample some images and display
    :param dataloader: maybe trainloader or testloader
    :param length: number of images to be displayed
    :param imgName: the name of saving image
    :return:
    """
    if params.fig_mode is None:
        return

    # randomly sample some images.
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    # process images so they can be displayed.
    images = images[:length]

    images = torchvision.utils.make_grid(images).numpy()
    images = images/2 + 0.5
    images = np.transpose(images, (1, 2, 0))


    if params.fig_mode == 'display':

        plt.imshow(images)
        plt.show()

    if params.fig_mode == 'save':
        # Check if folder exist, otherwise need to create it.
        folder = os.path.abspath(params.save_dir)

        if not os.path.exists(folder):
            os.makedirs(folder)

        if imgName is None:
            imgName = 'displayImages' + str(int(time.time()))


        # Check extension in case.
        if not (imgName.endswith('.jpg') or imgName.endswith('.png') or imgName.endswith('.jpeg')):
            imgName = os.path.join(folder, imgName + '.jpg')

        plt.imsave(imgName, images)
        plt.close()

    # print labels
    print(' '.join('%5s' % labels[j].item() for j in range(length)))

def plot_embedding(X, y, d, title=None, imgName=None):
    """
    Plot an embedding X with the class label y colored by the domain d.
    :param X: embedding
    :param y: label
    :param d: domain
    :param title: title on the figure
    :param imgName: the name of saving image
    :return:
    """
    if params.fig_mode is None:
        return

    # normalization
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i]/1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])

    # If title is not given, we assign training_mode to the title.
    if title is not None:
        plt.title(title)
    else:
        plt.title(params.training_mode)

    if params.fig_mode == 'display':
        # Directly display if no folder provided.
        plt.show()

    if params.fig_mode == 'save':
        # Check if folder exist, otherwise need to create it.
        folder = os.path.abspath(params.save_dir)

        if not os.path.exists(folder):
            os.makedirs(folder)

        if imgName is None:
            imgName = 'plot_embedding' + str(int(time.time()))

        # Check extension in case.
        if not (imgName.endswith('.jpg') or imgName.endswith('.png') or imgName.endswith('.jpeg')):
            imgName = os.path.join(folder, imgName + '.jpg')

        print('Saving ' + imgName + ' ...')
        plt.savefig(imgName)
        plt.close()