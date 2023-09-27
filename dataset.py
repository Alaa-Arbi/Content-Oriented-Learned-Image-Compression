import os
import glob 
import json
import math
import PIL
import numpy as np
from collections import namedtuple
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from generate_structure_masks import generate_masks
import yolov5_face.detect_face as df

class ContentOrientedDataset(Dataset):
    def __init__(self, root='', crop_size=256, 
        normalize=False, **kwargs):
        super().__init__()

        self.data_dir = root 
        img_extensions = ['.jpg', '.png']
        self.imgs = []
        for ext in img_extensions:
            self.imgs += glob.glob(os.path.join(self.data_dir, f'images/**/*{ext}'), recursive=True)
        self.crop_size = crop_size
        self.image_dims = (3, self.crop_size, self.crop_size)
        self.normalize = normalize
        
    
        json_file_path = os.path.join(self.data_dir, "face_coords.json")
        if not(os.path.exists(json_file_path)):
            self.preprocess()
        with open(json_file_path, 'r') as json_file:
            self.face_coords = json.load(json_file)

    def _augment(self, img, face_masks, structure_masks):
        """
        Apply augmentations 
        """
        SCALE_MIN = 0.75
        SCALE_MAX = 0.95
        H, W, _ = img.shape # slightly confusing
        shortest_side_length = min(H,W)
        minimum_scale_factor = float(self.crop_size) / float(shortest_side_length)
        scale_low = max(minimum_scale_factor, SCALE_MIN)
        scale_high = max(scale_low, SCALE_MAX)
        scale = np.random.uniform(scale_low, scale_high)

        self.augmentations = iaa.Sequential([iaa.Fliplr(0.5), # horizontally flip 50% of the images
                                             iaa.Resize((math.ceil(scale * H), math.ceil(scale * W))), # resize
                                             iaa.size.CropToFixedSize(self.crop_size,self.crop_size)])
        
        masks = np.dstack( [face_masks, structure_masks])
        masks = SegmentationMapsOnImage(masks, shape=(H,W,2))
        img, masks = self.augmentations(image=img, segmentation_maps=masks)
        masks = masks.get_arr()
        face_masks, structure_masks = masks[:,:,0], masks[:,:,1]
        
        return img, face_masks, structure_masks

    def _transforms(self, img, face_mask, structure_mask):
        """
        Create and apply transforms
        """
        to_tensor = transforms.ToTensor()
        transforms_list = [to_tensor]
        if self.normalize is True:
            transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        img = transforms.Compose(transforms_list)(img)

        face_mask, structure_mask = to_tensor(face_mask).type(torch.int8), to_tensor(structure_mask).type(torch.int8)
    
        return img, face_mask, structure_mask

    def get_face_mask(self, idx, shape): 
        """
        Create the face mask 
        """
        H,W,_ = shape
        mask = np.zeros((H, W, 1), dtype=np.int32)
        coords = self.face_coords[os.path.relpath(self.imgs[idx], os.path.join(self.data_dir,"images"))]
        for coord in coords:
             mask[coord[1]:coord[3],coord[0]:coord[2]] = 1 
        return mask 

    def get_structure_mask(self, idx):
        """
        Create the structure mask
        """
        rel_path = os.path.relpath(self.imgs[idx], os.path.join(self.data_dir,"images"))
        rel_path = os.path.splitext(rel_path)[0]+".png"
        mask = PIL.Image.open(os.path.join(self.data_dir, "structure_masks", rel_path))
        mask = np.expand_dims(np.array(mask), axis=-1)
        mask = mask / 255 
        mask = mask.astype(np.int8)
        return mask

    def preprocess(self):
        # detect all small faces and save them to face_coords.json file
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = "yolov5_face/weights/yolov5m-face.pt"
        model = df.load_model(weights, device)
        in_dir = os.path.join(self.data_dir, "images")
        out_file = os.path.join(self.data_dir, "face_coords.json")
        df.detect(model, in_dir, out_file)
        # create structure masks using canny edge detector
        config = namedtuple("MaskGenerationConfig",
                                    ["input_dir ",             
                                    "output_dir",         
                                    "min",      
                                    "max"])   
        config.input_dir = os.path.join(self.data_dir, "images")
        config.output_dir = os.path.join(self.data_dir, "structure_masks")
        config.min, config.max = 70, 200
        generate_masks(config)


    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        filesize = os.path.getsize(img_path)
        img = PIL.Image.open(img_path)
        img = img.convert('RGB') 
        img = np.array(img)
        H, W, _ = img.shape # slightly confusing
        bpp = filesize * 8. / (H * W)
        face_mask = self.get_face_mask(idx, img.shape)
        structure_mask = self.get_structure_mask(idx)
        img, face_mask, structure_mask = self._augment(img, face_mask, structure_mask)    
        img, face_mask, structure_mask = self._transforms(img, face_mask, structure_mask)
        return [img, face_mask, structure_mask], bpp

root = "dataset"
dataset = ContentOrientedDataset(root=root, crop_size=256, normalize=False)
print()

        