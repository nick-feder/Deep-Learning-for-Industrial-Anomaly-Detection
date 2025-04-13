import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

# fixed transform resize with yaml variable, data discovery with glob

class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train
    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        
        if self.is_train:
            return image
        else:
            h, w = image.shape[1], image.shape[2]
            
            if os.path.dirname(image_file).endswith("good"):

                target = torch.zeros(1, h, w)
            else:
                try:
                    mask_file = image_file.replace(f"{os.path.sep}test{os.path.sep}", 
                                                f"{os.path.sep}ground_truth{os.path.sep}").replace(
                                                ".png", "_mask.png")
                    mask = Image.open(mask_file)
                    mask = transforms.Resize((h, w))(mask)
                    mask = transforms.ToTensor()(mask)

                    target = (mask > 0).float()
                    if target.shape[0] != 1:

                        target = target[0:1]
                except Exception as e:
                    print(f"Error with mask {mask_file}: {e}")
                    target = torch.zeros(1, h, w)
                    
            return image, target

    def __len__(self):
        return len(self.image_files)