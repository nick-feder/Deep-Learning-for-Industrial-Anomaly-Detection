import os
from enum import Enum
import PIL
import torch
from torchvision import transforms
#adapted from https://github.com/amazon-science/patchcore-inspection/blob/main/src/patchcore/datasets/mvtec.py
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class MVTecMultiViewDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        source,
        classnames, 
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
 
        super().__init__()
        assert len(classnames) == 2, "Must provide exactly two subdatasets for multi-view input."
        
        self.source = source
        self.split = split
        self.classnames = classnames
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        
        self.transform_img = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])

        self.imagesize = (6, imagesize, imagesize) 

    def __getitem__(self, idx):
        (classname1, anomaly1, image_path1, mask_path1), (classname2, anomaly2, image_path2, mask_path2) = self.data_to_iterate[idx]
        
        image1 = PIL.Image.open(image_path1).convert("RGB")
        image1 = self.transform_img(image1)
        
        image2 = PIL.Image.open(image_path2).convert("RGB")
        image2 = self.transform_img(image2)
        
        image_concat = torch.cat([image1, image2], dim=0)

        if self.split == DatasetSplit.TEST and mask_path1 is not None:
            mask1 = PIL.Image.open(mask_path1)
            mask1 = self.transform_mask(mask1)
        else:
            mask1 = torch.zeros([1, *image1.size()[1:]])

        if self.split == DatasetSplit.TEST and mask_path2 is not None:
            mask2 = PIL.Image.open(mask_path2)
            mask2 = self.transform_mask(mask2)
        else:
            mask2 = torch.zeros([1, *image2.size()[1:]])
        mask = torch.max(mask1, mask2)  
        
        return {
            "image": image_concat,
            "mask": mask,
            "classname": classname1,  
            "anomaly": anomaly1,  
            "is_anomaly": int(anomaly1 != "good"),
            "image_name": (image_path1, image_path2),
            "image_path": (image_path1, image_path2),
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for i, classname in enumerate(self.classnames):
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None
        data_to_iterate = []
        for (classname1, classname2) in [(self.classnames[0], self.classnames[1])]:
            for anomaly in sorted(imgpaths_per_class[classname1].keys()):
                for i, image_path1 in enumerate(imgpaths_per_class[classname1][anomaly]):
                    image_path2 = imgpaths_per_class[classname2][anomaly][i] 
                    mask_path1 = maskpaths_per_class[classname1][anomaly][i] if maskpaths_per_class[classname1][anomaly] else None
                    mask_path2 = maskpaths_per_class[classname2][anomaly][i] if maskpaths_per_class[classname2][anomaly] else None
                    
                    data_to_iterate.append([
                        (classname1, anomaly, image_path1, mask_path1),
                        (classname2, anomaly, image_path2, mask_path2)
                    ])
        
        return imgpaths_per_class, data_to_iterate