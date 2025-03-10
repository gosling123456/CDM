import cv2
import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from improved_diffusion.image_datasets import _list_image_files_recursively
from improved_diffusion.dist_util import dev


def make_transform(model_type: str, resolution: int):
    """ Define input transforms for pretrained models """
    if model_type == 'ddpm':
        transform = transforms.Compose([
            # transforms.Resize(resolution),
            transforms.ToTensor(),
            lambda x: 2 * x - 1
        ])
    elif model_type in ['mae', 'swav', 'swav_w2', 'deeplab']:
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return transform


class FeatureDataset(Dataset):
    ''' 
    Dataset of the pixel representations and their labels.

    :param X_data: pixel representations [num_pixels, feature_dim]
    :param y_data: pixel labels [num_pixels]
    '''
    def __init__(
        self, 
        X_data: torch.Tensor, 
        y_data: torch.Tensor
    ):    
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class ImageLabelDataset(Dataset):
    ''' 
    :param data_dir: path to a folder with images and their annotations. 
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''
    def __init__(
        self,
        data_dir: str,
        resolution: int,
        num_images= -1,
        transform=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.transform = transform
        self.image_paths = _list_image_files_recursively(data_dir)
        self.image_paths = sorted(self.image_paths)

        if num_images > 0:
            print(f"Take first {num_images} images...")
            self.image_paths = self.image_paths[:num_images]

        self.label_paths = [
            '.'.join(image_path.split('.')[:-1] + ['npy'])
            for image_path in self.image_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image
        image_path = self.image_paths[idx]
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")
        assert pil_image.size[0] == pil_image.size[1], \
               f"Only square images are supported: ({pil_image.size[0]}, {pil_image.size[1]})"

        tensor_image = self.transform(pil_image)
        # Load a corresponding mask and resize it to (self.resolution, self.resolution)
        label_path = self.label_paths[idx]
        label = np.load(label_path).astype('uint8')
        label = cv2.resize(
            label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST
        )
        tensor_label = torch.from_numpy(label)
        return tensor_image, tensor_label
    
class SARImageLabelDataset(Dataset):
    def __init__(self, root, list_path, transform = None, ignore_label=255):
        self.root = root                        
        self.list_path = list_path                   
        self.ignore_label = ignore_label        
        self.img_ids = [i_id.strip() for i_id in open(list_path)]  
        self.transform = transform

        self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}


    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(os.path.join(self.root, str("JPEGImages/%s" % name) + ".jpg" )).convert('RGB')
        label = Image.open(os.path.join(self.root, str("SegmentationClass/%s" % name) + ".png")).convert('P')

        image_np = np.asarray(image) 
        label = np.asarray(label).astype('uint8')

        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)
        
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        # 特征可视化 增加  
        # image_np = Image.fromarray(image_np)

        return self.transform(image_np.copy()), torch.Tensor(label_copy.copy())

class AE_SARImageLabelDataset(Dataset):
    def __init__(self, args, root, list_path, feature_extractor, collect_features, noise, transform = None, ignore_label=255):
        self.args = args
        self.root = root                        
        self.list_path = list_path                   
        self.ignore_label = ignore_label        
        self.img_ids = [i_id.strip() for i_id in open(list_path)]  
        self.transform = transform
        self.feature_extractor = feature_extractor  
        self.collect_features = collect_features
        self.noise = noise
        self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}


    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(os.path.join(self.root, str("JPEGImages/%s" % name) + ".jpg" )).convert('RGB')  
        image_np = np.asarray(image) 
        out = self.transform(image_np.copy())[None].to(dev())
        features = self.feature_extractor(out, noise=self.noise) 
        out = self.collect_features(self.args, features).cpu()     

        return out

class InMemoryImageLabelDataset(Dataset):
    ''' 

    Same as ImageLabelDataset but images and labels are already loaded into RAM.
    It handles DDPM/GAN-produced datasets and is used to train DeepLabV3. 

    :param images: np.array of image samples [num_images, H, W, 3].
    :param labels: np.array of correspoding masks [num_images, H, W].
    :param resolution: image and mask output resolusion.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''

    def __init__(
            self,
            images: np.ndarray, 
            labels: np.ndarray,
            resolution=256,
            transform=None
    ):
        super().__init__()
        assert  len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        assert image.size[0] == image.size[1], \
               f"Only square images are supported: ({image.size[0]}, {image.size[1]})"

        tensor_image = self.transform(image)
        label = self.labels[idx]
        label = cv2.resize(
            label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST
        )
        tensor_label = torch.from_numpy(label)
        return tensor_image, tensor_label

# def label_adjust():
    # 

class SARnpyLabelDataset(Dataset):
    def __init__(self, root, list_path, transform = None, ignore_label=255):
        self.root = root                        
        self.list_path = list_path                   
        self.ignore_label = ignore_label        
        self.img_ids = [i_id.strip() for i_id in open(list_path)]  
        self.transform = transform

        # self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}


    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        name = self.img_ids[index]
        sar_num = name.split('/')[0]
        
        image_path = os.path.join(self.root, "feature/" + name)
        label_path = os.path.join(self.root, "label/" + name)

        image_np = np.load(image_path).transpose((1,2,0))
        label = np.load(label_path)
        # print(image_np.shape, label.shape)

        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)
        if sar_num == 'sar4':
            id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 4, 4: 4}
        elif sar_num == 'sar13' or sar_num == 'sar14' or sar_num == 'sar15':
            id_to_trainid = {0: 4, 1: 2, 2: 0, 3: 3, 4: 1, 5: 4}
        else:
            id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        
        for k, v in id_to_trainid.items():
            label_copy[label == k] = v

        return self.transform(image_np.copy()), torch.Tensor(label_copy.copy())
        
class SARnpyTestLabelDataset(Dataset):
    def __init__(self, root, list_path, sar_num, transform = None, ignore_label=255):
        self.root = root                        
        self.list_path = list_path                   
        self.ignore_label = ignore_label        
        self.img_ids = [i_id.strip() for i_id in open(list_path)]  
        self.transform = transform
        if sar_num == 'sar4':
            self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 4, 4: 4}
        elif sar_num == 'sar13' or sar_num == 'sar14' or sar_num == 'sar15':
            self.id_to_trainid = {0: 4, 1: 2, 2: 0, 3: 3, 4: 1, 5: 4}
        else:
            self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}


    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        name = self.img_ids[index]
        image = Image.open(os.path.join(self.root, str("JPEGImages/%s" % name) + ".jpg" )).convert('RGB')
        label = Image.open(os.path.join(self.root, str("SegmentationClass/%s" % name) + ".png")).convert('P')

        image_np = np.asarray(image)
        label = np.asarray(label).astype('uint8')
        # print(image_np.shape)

        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)
        
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        return self.transform(image_np.copy()), torch.Tensor(label_copy.copy())

class newSARImageLabelDataset(Dataset):
    def __init__(self, root, list_path, transform = None, ignore_label=255):
        self.root = root                        
        self.list_path = list_path                   
        self.ignore_label = ignore_label        
        self.img_ids = [i_id.strip() for i_id in open(list_path)]  
        self.transform = transform


    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        name = self.img_ids[index]
        idx_num = name.split('_')[0]
        #print(idx_num)
        image = Image.open(os.path.join(self.root, str("image/%s" % name) + ".jpg" )).convert('RGB')
        label = Image.open(os.path.join(self.root, str("label/%s" % name) + ".png")).convert('P')

        image_np = np.asarray(image)
        label = np.asarray(label).astype('uint8')
        #print(np.max(label))

        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)
        if idx_num == '4':
            id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 4, 4: 4}
        elif idx_num == '13' or idx_num == '14' or idx_num == '15':
            id_to_trainid = {0: 4, 1: 2, 2: 0, 3: 3, 4: 1, 5: 4}
        else:
            id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        for k, v in id_to_trainid.items():
            label_copy[label == k] = v
        
        return self.transform(image_np.copy()), torch.Tensor(label_copy.copy())
