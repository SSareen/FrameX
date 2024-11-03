import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

#function to prepare data by downsampling high-res images
def prepare_data(train_high_res_dir, val_high_res_dir, working_dir):
    os.makedirs(os.path.join(working_dir, 'train', 'inputs'), exist_ok=True)
    os.makedirs(os.path.join(working_dir, 'val', 'inputs'), exist_ok=True)

    #prepare training data
    print("Preparing Training Data")
    for dirname, _, filenames in os.walk(train_high_res_dir):
        for filename in filenames:
            high_res_path = os.path.join(dirname, filename)
            label = cv2.imread(high_res_path)
            if label is None:
                print(f"Warning: Unable to read image {high_res_path}. Skipping.")
                continue
            input_img = cv2.resize(label, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
            low_res_path = os.path.join(working_dir, 'train', 'inputs', 'in_' + filename)
            cv2.imwrite(low_res_path, input_img)

    #prepare validation data
    print("Preparing Validation Data")
    for dirname, _, filenames in os.walk(val_high_res_dir):
        for filename in filenames:
            high_res_path = os.path.join(dirname, filename)
            label = cv2.imread(high_res_path)
            if label is None:
                print(f"Warning: Unable to read image {high_res_path}. Skipping.")
                continue
            input_img = cv2.resize(label, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
            low_res_path = os.path.join(working_dir, 'val', 'inputs', 'in_' + filename)
            cv2.imwrite(low_res_path, input_img)

#custom dataset class for super-resolution
class SRDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        #list all low-res images
        self.low_res_images = sorted([
            x for x in os.listdir(low_res_dir) 
            if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
        ])
        self.transform = transform

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        low_res_filename = self.low_res_images[idx]
        #remove 'in_' prefix to get the high-res filename
        high_res_filename = low_res_filename[3:]

        low_res_img_path = os.path.join(self.low_res_dir, low_res_filename)
        high_res_img_path = os.path.join(self.high_res_dir, high_res_filename)

        #check if high-res image exists
        if not os.path.exists(high_res_img_path):
            raise FileNotFoundError(f"High-res image {high_res_img_path} not found.")

        #open images
        low_res_image = Image.open(low_res_img_path).convert('RGB')
        high_res_image = Image.open(high_res_img_path).convert('RGB')

        #upsample low-res image to high-res size using bicubic interpolation
        upsampled_image = low_res_image.resize(high_res_image.size, Image.BICUBIC)

        if self.transform:
            upsampled_image = self.transform(upsampled_image)
            high_res_image = self.transform(high_res_image)

        return upsampled_image, high_res_image
