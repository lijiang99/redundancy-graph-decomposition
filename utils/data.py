from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def load_cifar10(path, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_set = datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_set = datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader
