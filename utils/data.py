import os
import pickle
import urllib
import tarfile
import PIL.Image
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.model_zoo import tqdm
 
def load_cifar10(path, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    
    train_set = datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_set = datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def load_cifar100(path, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    
    train_set = datasets.CIFAR100(root=path, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_set = datasets.CIFAR100(root=path, train=False, download=True, transform=transform_test)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader

def load_imagenet(path, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_set = datasets.ImageFolder(root=os.path.join(path, "train"), transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_set = datasets.ImageFolder(root=os.path.join(path, "val"), transform=transform_val)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

class CUB200(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self._raw_path = os.path.join(root, "raw")
        self._processed_path = os.path.join(root, "processed")
        self._train = train
        self._transform = transform
        self._target_transform = target_transform

        if self._checkIntegrity():
            print("Files already downloaded and verified")
        elif download:
            url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
            self._download(url)
            self._extract(self._processed_path)
        else:
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        
        if self._train:
            self._train_data, self._train_labels = pickle.load(open(os.path.join(self._processed_path, "train.pkl"), "rb"))
        else:
            self._test_data, self._test_labels = pickle.load(open(os.path.join(self._processed_path, "test.pkl"), "rb"))
    
    def __getitem__(self, index):
        if self._train:
            image, target = self._train_data[index], self._train_labels[index]
        else:
            image, target = self._test_data[index], self._test_labels[index]
        image = PIL.Image.fromarray(image)

        if self._transform is not None:
            image = self._transform(image)
        if self._target_transform is not None:
            target = self._target_transform(target)

        return image, target

    def __len__(self):
        return len(self._train_data) if self._train else len(self._test_data)

    def _checkIntegrity(self):
        return (os.path.isfile(os.path.join(self._processed_path, "train.pkl"))
            and os.path.isfile(os.path.join(self._processed_path, "test.pkl")))
    
    def _urlretrieve(self, url, filename, chunk_size=1024*32):
        with urllib.request.urlopen(urllib.request.Request(url)) as response:
            with open(filename, "wb") as fh, tqdm(total=response.length) as pbar:
                while chunk := response.read(chunk_size):
                    fh.write(chunk)
                    pbar.update(len(chunk))
        return
    
    def _download(self, url):
        if not os.path.isdir(self._raw_path):
            os.makedirs(self._raw_path)
        if not os.path.isdir(self._processed_path):
            os.makedirs(self._processed_path)
        
        # downloads file.
        fpath = os.path.join(self._raw_path, "CUB_200_2011.tgz")
        try:
            print("Downloading " + url + " to " + fpath)
            self._urlretrieve(url, fpath)
        except (urllib.error.URLError, OSError) as e:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print("Failed download. Trying https -> http instead. Downloading " + url + " to " + fpath)
                self._urlretrieve(url, fpath)
            else:
                raise e
        
        # extract file.
        cwd = os.getcwd()
        tar = tarfile.open(fpath, "r:gz")
        os.chdir(self._raw_path)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        return
    
    def _extract(self, extract_root):
        image_path = os.path.join(self._raw_path, "CUB_200_2011/images/")
        # format of images.txt: <image_id> <image_name>
        id2name = np.genfromtxt(os.path.join(self._raw_path, "CUB_200_2011/images.txt"), dtype=str)
        # format of train_test_split.txt: <image_id> <is_training_image>
        id2train = np.genfromtxt(os.path.join(self._raw_path, "CUB_200_2011/train_test_split.txt"), dtype=int)
        
        archive = os.path.join(self._raw_path, "CUB_200_2011.tgz")
        print(f"Extracting {archive} to {extract_root}")
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for id_ in range(id2name.shape[0]):
            image = PIL.Image.open(os.path.join(image_path, id2name[id_, 1]))
            label = int(id2name[id_, 1][:3]) - 1

            # convert gray scale image to rgb image.
            if image.getbands()[0] == "L":
                image = image.convert("RGB")
            image_np = np.array(image)
            image.close()

            if id2train[id_, 1] == 1:
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_data.append(image_np)
                test_labels.append(label)

        pickle.dump((train_data, train_labels), open(os.path.join(self._processed_path, "train.pkl"), "wb"))
        pickle.dump((test_data, test_labels), open(os.path.join(self._processed_path, "test.pkl"), "wb"))
        return

def load_cub200(path, batch_size):
    transform_train = transforms.Compose([
        transforms.Resize(size=448),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=448),
        transforms.CenterCrop(size=448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_set = CUB200(root=path, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_set = CUB200(root=path, train=False, download=True, transform=transform_test)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader