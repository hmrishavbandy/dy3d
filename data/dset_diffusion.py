import torch
import numpy as np
import os
import json
import PIL
from PIL import Image


class ShapeNet_Diffuse(torch.utils.data.dataset.Dataset):
    def __init__(self, 
                 base_dir="/mnt/linux_store/ldm_final/",
                 dset_dir = "/mnt/linux_store/Chair_Processed/03001627/", 
                 train=True, 
                 inversions=True, view_ = '3', 
                 preprocessor=None):
        self.preprocessor = preprocessor
        self.train = train
        self.view_ = view_        
        
        self.inversion = inversions
        self.root_dir = dset_dir
        self.base_dir = base_dir
        
        self.data = []
        self.load_data()
        print("Discovered {} sketches".format(len(self.data)))
        

        if inversions:
            fname = "data/lists/chairs_list.txt"
            with open(fname, "r") as f:
                f1 = f.readlines()
            f1 = [f[:-1] for f in f1]
            
            ar_ = np.load(f"{base_dir}/samples_final.npy")
            
            ar_ = {f1[i]:ar_[i] for i in range(len(ar_))}

            

            self.inversion_ar = ar_

        if train:
            fname = "data/lists/sv2_chairs_train.json"
            with open(fname, "r") as f:
                self.file_list = json.load(f)['ShapeNetV2']['03001627']

            fname = "data/lists/sv2_chairs_test.json"
            with open(fname, "r") as f:
                self.non_file_list = json.load(f)['ShapeNetV2']['03001627']
            
            self.file_list = self.file_list+[f for f in f1 if f not in self.non_file_list]
            
            print("Train List has : ", len(self.file_list), " files")
        else:
            fname = "data/lists/sv2_chairs_test.json"
            with open(fname, "r") as f:
                self.file_list = json.load(f)['ShapeNetV2']['03001627']
            fname = "data/lists/sv2_chairs_train.json"
            
            print("Test List has : ", len(self.file_list), " files")

        if inversions:
            for f in self.file_list:
                if f not in self.inversion_ar.keys():
                    self.file_list.remove(f)
            
        self.fname = fname

        tmp_data = []
        for file in self.data:
            f = file.split('/')[-2]
            if f in self.file_list:
                tmp_data.append(file)
        
        self.data = tmp_data
        self.tax_list = list(set([i.split('/')[-2] for i in self.data]))
        self.tax_list.sort()
        self.tax_dict = {self.tax_list[i]:i for i in range(len(self.tax_list))}

        if train:
            print('Total number of sketches in train : ', len(self.data))
        else:
            print('Total number of sketches in test : ', len(self.data))
    
    def label_to_id(self, label):
        return [self.tax_dict[l] for l in label]

    def __len__(self):
        return len(self.data)
    
    def load_data(self):
        print(self.root_dir)
        for files in os.listdir(self.root_dir):
            for file in os.listdir(os.path.join(self.root_dir, files)):
                if self.view_ != 'all':
                    if file.endswith('.png') and int(file.split('_')[-1].split(".")[0]) in self.view_ and file.split('_')[0] != 'prosketch':
                        self.data.append(os.path.join(self.root_dir,files, file))
                else:
                    if file.endswith('.png') and file.split('_')[0] != 'prosketch':
                        self.data.append(os.path.join(self.root_dir,files, file))
        
        
    def auto_crop_and_pad(self,img):
        img = img.convert('L')
        # Resize to 512x512
        img = img.resize((512, 512), Image.BICUBIC)

        # Convert the image to a NumPy array
        img_array = np.array(img)
        if np.max(img_array) == 1:
            img_array = img_array*255
        
        # Find the coordinates of the sketch pixels
        sketch_coords = np.argwhere(img_array >=128)

        # Calculate the bounding box of the sketch
        min_y, min_x = np.min(sketch_coords, axis=0)
        max_y, max_x = np.max(sketch_coords, axis=0)

        # Crop the image to the bounding box
        cropped_img = img.crop((min_x-5, min_y-5, max_x + 5, max_y + 5))

        # Find the minimum of height and width
        max_size = max(cropped_img.width, cropped_img.height)

        # Create a new square image with a white background
        square_img = Image.new('RGB', (max_size, max_size), color='black')

        # Calculate the padding values
        pad_x = (max_size - cropped_img.width) // 2
        pad_y = (max_size - cropped_img.height) // 2

        # Paste the cropped image onto the square canvas
        square_img.paste(cropped_img, (pad_x, pad_y))

        # Save the result
        return square_img
        
    def __getitem__(self, idx):
        
        img = PIL.Image.open(self.data[idx]).convert('RGB')

        img = self.preprocessor(img)
        f_name = self.data[idx].split('/')[-2]
        if not self.inversion:
            return img, f_name

        init_file = self.inversion_ar[f_name]
        init_file = torch.tensor(init_file)
         
        taxonomy_name = f_name
        return img, init_file, taxonomy_name 
