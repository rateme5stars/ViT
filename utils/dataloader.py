import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, dataset_paths, meta_file, transform=None):
        self.dataset_paths = dataset_paths
        self.transform = transform
        self.samples = []

        _cls_idx = self.class_to_idx()

        with open(meta_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                image_class, image_name = line.strip().split('/')
                image_path = os.path.join(dataset_paths, 'images', image_class, image_name + '.jpg')
                label = _cls_idx[image_class]
                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def class_to_idx(self):
        class_idx_map = {}
        for i, cls in enumerate(os.listdir(self.dataset_paths + '/images')):
            class_idx_map[cls] = i
        return class_idx_map
    
    def classes(self):
        return list(os.listdir(self.dataset_paths + '/images'))

def create_dataloader(batch_size: int):
    train_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomPerspective(),
    transforms.RandomAffine(degrees=(30, 70)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    dataset_paths = os.path.join(parent_dir, 'data/food-101')

    train_meta_file = os.path.join(dataset_paths, 'meta', 'train.txt')
    test_meta_file = os.path.join(dataset_paths, 'meta', 'test.txt')

    train_dataset = CustomDataset(dataset_paths, train_meta_file, transform=train_transforms)
    test_dataset = CustomDataset(dataset_paths, test_meta_file, transform=test_transforms)  
    num_classes = train_dataset.classes()

    num_workers = os.cpu_count() - 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, num_classes

