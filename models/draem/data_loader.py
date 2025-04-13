import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
##
# added path error handling
# added ram preloading for anomalous images in dataset init to speed up training
##

class MVTecDRAEMTestDataset(Dataset):
    def __init__(self, root_dir, resize_shape=None):
        print(f"Initializing dataset with root_dir: {root_dir}")
        self.root_dir = os.path.abspath(root_dir)

        if not os.path.exists(self.root_dir):
            print(f"ERROR: Dataset folder does NOT exist: {self.root_dir}")
        else:
            print(f" Dataset folder exists: {self.root_dir}")

        self.image_paths = sorted(
                glob.glob(os.path.join(self.root_dir, "**", "*.jpg"), recursive=True) +
                glob.glob(os.path.join(self.root_dir, "**", "*.png"), recursive=True)
            )

        if len(self.image_paths) == 0:
            print(f" No images found in {self.root_dir}!")
        self.image_paths = [os.path.abspath(path) for path in self.image_paths]

        print(f"Found {len(self.image_paths)} images in {self.root_dir}.")

        self.resize_shape = resize_shape


    def __len__(self):
        return len(self.image_paths)

    def transform_image(self, image_path, mask_path):

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Failed to load image from {image_path}")

        if mask_path is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"ailed to load mask from {mask_path}, using a blank mask instead.")
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            print(f"Mask file not found for {image_path}, using a blank mask.")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if self.resize_shape is not None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0


        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path) 
        
        if base_dir == 'good':
            image, _ = self.transform_image(img_path, None)
            mask = np.zeros((1, self.resize_shape[0], self.resize_shape[1]), dtype=np.float32)  # Empty mask
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            file_base = os.path.splitext(file_name)[0] 
            mask_path = os.path.abspath(os.path.join(
                os.path.dirname(self.root_dir), 
                "ground_truth", 
                base_dir,
                f"{file_base}_mask.png" 
            ))
            if not os.path.exists(mask_path):
                print(f" WARNING: Mask not found at {mask_path}. Using empty mask.")
                image, _ = self.transform_image(img_path, None)
                mask = np.zeros((1, self.resize_shape[0], self.resize_shape[1]), dtype=np.float32)
            else:
                image, mask = self.transform_image(img_path, mask_path)
                
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly, 'mask': mask, 'idx': idx}
        return sample

class MVTecDRAEMTrainDataset(Dataset):
    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            anomaly_source_path (string): Directory with anomaly images.
            resize_shape (tuple): Resize images to this shape.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape

        self.image_paths = sorted(
            glob.glob(os.path.join(root_dir, "*.jpg"), recursive=True) +
            glob.glob(os.path.join(root_dir, "*.png"), recursive=True)
        )

        self.anomaly_source_paths = sorted(
            glob.glob(os.path.join(anomaly_source_path, "*/*.jpg")) +
            glob.glob(os.path.join(anomaly_source_path, "*/*.png"))
        )

        print(f"Preloading {len(self.image_paths)} normal images into RAM")
        self.image_cache = {p: cv2.resize(cv2.imread(p), (resize_shape[1], resize_shape[0])) for p in self.image_paths}

        print(f"Preloading {len(self.anomaly_source_paths)} anomaly images into RAM")
        self.anomaly_cache = {p: cv2.resize(cv2.imread(p), (resize_shape[1], resize_shape[0])) for p in self.anomaly_source_paths}

        print(f" Done, loaded {len(self.image_cache)} normal images and {len(self.anomaly_cache)} anomaly images.")

        # Augmenters
        self.augmenters = [
            iaa.GammaContrast((0.5,2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2), add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50,50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32,128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        return iaa.Sequential([self.augmenters[aug_ind[0]], self.augmenters[aug_ind[1]], self.augmenters[aug_ind[2]]])

    def augment_image(self, image, anomaly_image):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0

        anomaly_img_augmented = aug(image=anomaly_image)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * perlin_thr

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            return image.astype(np.float32), np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = perlin_thr.astype(np.float32)
            augmented_image = msk * augmented_image + (1-msk) * image
            has_anomaly = np.array([1.0], dtype=np.float32) if np.sum(msk) > 0 else np.array([0.0], dtype=np.float32)
            return augmented_image, msk, has_anomaly

    def transform_image(self, image, anomaly_image):
        if torch.rand(1).item() > 0.7:
            image = self.rot(image=image)
        image = image.astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_image)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))

        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        anomaly_path = np.random.choice(self.anomaly_source_paths)
        image = self.image_cache[image_path]
        anomaly_image = self.anomaly_cache[anomaly_path]
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(image, anomaly_image)

        return {
            'image': image,
            'augmented_image': augmented_image,
            'anomaly_mask': anomaly_mask,
            'has_anomaly': has_anomaly,
            'idx': idx
        }