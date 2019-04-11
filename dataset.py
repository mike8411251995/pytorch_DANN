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


def LoadDataset(name, root, batch_size, split,shuffle=True, style=None, attr=None):
    if name == 'mnist':
        if split == 'train':
            return LoadMNIST(root+'mnist/', batch_size=batch_size, split='train', shuffle=shuffle, scale_32=True)
        elif split=='test':
            return LoadMNIST(root+'mnist/', batch_size=batch_size, split='test', shuffle=False, scale_32=True)
    elif name == 'usps':
        if split == 'train':
            return LoadUSPS(root+'usps/', batch_size=batch_size, split='train', shuffle=shuffle, scale_32=True)
        elif split=='test':
            return LoadUSPS(root+'usps/', batch_size=batch_size, split='test', shuffle=False, scale_32=True)
    elif name == 'svhn':
        if split == 'train':
            return LoadSVHN(root+'svhn/', batch_size=batch_size, split='extra', shuffle=shuffle)
        elif split=='test':
            return LoadSVHN(root+'svhn/', batch_size=batch_size, split='test', shuffle=False)
    elif name == 'mnistm':
        if split == 'train':
            return LoadMNISTM(root+'mnistm/', batch_size=batch_size, split='train', shuffle=shuffle, scale_32=True)
        elif split=='test':
            return LoadMNISTM(root+'mnistm/', batch_size=batch_size, split='test', shuffle=False, scale_32=True)
    # elif name == 'face':
    #     assert style != None
    #     if split == 'train':
    #         return LoadFace(root, style=style, split='train', batch_size=batch_size,  shuffle=shuffle)
    #     elif split=='test':
    #         return LoadFace(root, style=style, split='test', batch_size=batch_size,  shuffle=shuffle)


def LoadSVHN(data_root, batch_size=32, split='train', shuffle=True):
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    svhn_dataset = datasets.SVHN(data_root, split=split, download=True,
                                   transform=transforms.ToTensor())
    return DataLoader(svhn_dataset,batch_size=batch_size, shuffle=shuffle, drop_last=True)

def LoadUSPS(data_root, batch_size=32, split='train', shuffle=True, scale_32 = False):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    usps_dataset = USPS_2(root=data_root,train=(split=='train'),download=True,scale_32=scale_32)
    return DataLoader(usps_dataset,batch_size=batch_size, shuffle=shuffle, drop_last=True)

def LoadMNIST(data_root, batch_size=32, split='train', shuffle=True, scale_32 = False):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if scale_32:
        trans = transforms.Compose([transforms.Resize(size=[32, 32]),transforms.ToTensor()])
    else:
        trans = transforms.ToTensor()

    mnist_dataset = datasets.MNIST(data_root, train=(split=='train'), download=True,
                                   transform=trans)
    return DataLoader(mnist_dataset,batch_size=batch_size,shuffle=shuffle, drop_last=True)

def LoadMNISTM(data_root, batch_size=32, split='train', shuffle=True, scale_32 = False):
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if scale_32:
        trans = transforms.Compose([transforms.Resize(size=[32, 32]),transforms.ToTensor()])
    else:
        trans = transforms.ToTensor()

    mnist_dataset = MNISTM(data_root, train=(split=='train'), download=True,
                                   transform=trans)
    return DataLoader(mnist_dataset,batch_size=batch_size,shuffle=shuffle, drop_last=True)


def LoadFace(data_root, batch_size=32, split='train', style='photo', attr = None,
               shuffle=True, load_first_n = None):

    data_root = data_root+'face.h5'
    key = '/'.join(['CelebA',split,style])
    celeba_dataset = Face(data_root,key,load_first_n)
    return DataLoader(celeba_dataset,batch_size=batch_size,shuffle=shuffle,drop_last=True)


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
    
    def __init__(self, root, train=True, scale_32=False, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)

        if scale_32:
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

class Face(Dataset):
    def __init__(self, root, key, load_first_n = None):

        with h5py.File(root,'r') as f:
            data = f[key][()]
            if load_first_n:
                data = data[:load_first_n]
        self.imgs = (data/255.0)*2 -1

    def __getitem__(self, index):
        return self.imgs[index]

    def __len__(self):
        return len(self.imgs)


class dataset_unpair(Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot

    # A
    images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
    self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]

    # B
    images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
    self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    # transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size


class dataset_shoes(Dataset):
  def __init__(self, opts, phase='train', input_dim=3):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, phase))
    self.img_A = [os.path.join(self.dataroot, phase, x) for x in images]
    self.size = len(self.img_A)
    self.input_dim = input_dim
    self.img_B = [os.path.join(self.dataroot, phase, x) for x in images]
    self.pair = opts.pair
    if self.pair == 'False':
        random.shuffle(self.img_B)

    # setup image transformation
    transforms = [Resize((opts.crop_size, opts.crop_size*2), Image.BICUBIC)]
    #transforms.append(CenterCrop(opts.crop_size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('edge2shoes: %d images'%(self.size))
    return

  def __getitem__(self, index):
    if self.pair == 'False':
        data_A = self.load_img(img_name=self.img_A[random.randint(0, self.size - 1)], domain='A', input_dim=self.input_dim)
        data_B = self.load_img(img_name=self.img_B[random.randint(0, self.size - 1)], domain='B', input_dim=self.input_dim)
    elif self.pair == 'True':
        data_A = self.load_img(img_name=self.img_A[index], domain='A', input_dim=self.input_dim)
        data_B = self.load_img(img_name=self.img_B[index], domain='B', input_dim=self.input_dim)

    return data_A, data_B

  def load_img(self, img_name, domain, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
        img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
        img = img.unsqueeze(0)
    if domain == 'A':
        img_out = img[:,:,:64]
    else:
        img_out = img[:,:,64:]
    return img_out

  def __len__(self):
    return self.size


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
            download=False,scale_32=False):

        transform = []
        

        transform += [transforms.Resize(size=[32, 32]),transforms.ToTensor()]

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

class CelebADataset(Dataset):
    """docstring for MyDataset"""
    def __init__(self, mode, train_filepath, test_filepath, train_csvpath, test_csvpath):
        #self.train_data = read_image(train_filepath)
        #self.test_data = read_image(test_filepath)     
        self.mode = mode

        if self.mode == 'VAE_train':
            self.images = read_image(train_filepath)
        if self.mode == 'VAE_test':
            self.images = read_image(test_filepath)
        if self.mode == 'GAN':
            self.images = read_image_gan([train_filepath, test_filepath], True)

        if self.mode == 'ACGAN':
            self.images = read_image_gan([train_filepath, test_filepath], False)
            self.train_attr = pd.read_csv(train_csvpath)['Smiling']
            self.test_attr = pd.read_csv(test_csvpath)['Smiling']
            self.attr = pd.concat((self.train_attr, self.test_attr), ignore_index=True)
            #self.attr = torch.FloatTensor(self.attr)
            self.attr = torch.FloatTensor(self.attr).view(-1, 1, 1, 1)

        self.images = torch.FloatTensor(self.images)

    def __getitem__(self, index):
        data = self.images[index]
        if self.mode == 'ACGAN':
            label = self.attr[index]
            return data, label

        else: return data, data

    def __len__(self):
        return len(self.images)