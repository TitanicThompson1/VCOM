from PIL import Image
import PIL
import torch
from torch.utils.data import Dataset, DataLoader
import utils
import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from os.path import join

labels = {
    'trafficlight': 0,
    'stop': 1,
    'speedlimit': 2,
    'crosswalk': 3
}

NUMBER_CLASSES = 4

class TrafficSignDataset(Dataset):
    def __init__(self, img_dir, ann_dir, conf, transform=None):
        self.annotations = utils.parse_annotations(ann_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.multi_label = conf['multilabel']
        self.object_detection = conf['objectdetection']
        self.multi_class = conf['multiclass']
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        filename = self.annotations[idx]['filename']
        img = Image.open(self.img_dir + filename).convert("RGB")
        label = None

        if self.transform:
            img = self.transform(img)
        
        if self.multi_label:
            signs = utils.get_labels(self.annotations[idx]['objects'])
            label = torch.zeros(NUMBER_CLASSES)
            for sign in signs:
                label[labels[sign]] = 1
            
        elif self.multi_class:
            biggest_sign = utils.get_biggest_sign(self.annotations[idx]['objects'])
            label = labels[biggest_sign]
        elif self.object_detection:
            boxes = utils.get_boxes(self.annotations[idx]['objects'])
            temp_labels = torch.zeros(len(self.annotations[idx]['objects']))
            for i, obj in enumerate(self.annotations[idx]['objects']):
                temp_labels[i]= labels[obj['name']]
            
            label={}
            label['boxes'] = boxes
            label['labels'] = temp_labels
            label['image_id'] = torch.tensor(idx, dtype=torch.int64)
            label['area'] = utils.get_areas(self.annotations[idx]['objects'])

        return img, label




def get_loader(config):

    # data augmentation
    data_transf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20, interpolation=PIL.Image.BILINEAR),
            transforms.GaussianBlur(kernel_size=(3,3)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # data augmentation
    data_transf_val = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    training_data = TrafficSignDataset(join(utils.SRC_DIR, "dataset/train/"), 
                                    join(utils.SRC_DIR, "dataset/ann_train/"),
                                    config,
                                    transform=data_transf)

    val_data = TrafficSignDataset(join(utils.SRC_DIR, "dataset/train/"), 
                                    join(utils.SRC_DIR, "dataset/ann_train/"),
                                    config,
                                    transform=data_transf_val)
    
    indices = list(range(len(training_data)))
    np.random.shuffle(indices, )

    val_size = 0.15 * len(indices)
    split = int(np.floor(val_size))
    train_idx, val_idx = indices[split:], indices[:split]

    val_sampler = SubsetRandomSampler(val_idx)
    train_sampler = SubsetRandomSampler(train_idx)

    train_dataloader = DataLoader(training_data, sampler=train_sampler, batch_size=config['batchsize'], num_workers=config['numworkers'], drop_last=True)
    validation_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=config['batchsize'], num_workers=config['numworkers'])

    return train_dataloader, validation_dataloader

def get_test_loader(config):

    # data augmentation
    data_transf = None
    if config['objectdetection']:
        data_transf = transforms.Compose([
                transforms.ToTensor(),
        ])
        
    else:
        data_transf = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    test_data = TrafficSignDataset(join(utils.SRC_DIR, "dataset/test/"), 
                                join(utils.SRC_DIR, "dataset/ann_test/"),
                                config,
                                transform=data_transf)

    test_dataloader = DataLoader(test_data, batch_size=config['batchsize'], shuffle=False, num_workers=config['numworkers'])   

    return test_dataloader
    