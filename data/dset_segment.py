import torch
import os
import json
from PIL import Image



class ShapeNet_Segment(torch.utils.data.dataset.Dataset):
    def __init__(self, 
                root,
                segment_dir,
                train=True, inversions=True, view_ = '3' , preprocessor = None):
        
        self.train = train
        self.preprocessor = preprocessor
        self.inversion = inversions
        self.root_dir = root
        self.segment_dir = segment_dir
        self.data = []
        self.view_ = view_
        
        self.load_data()
        print("Discovered {} sketches".format(len(self.data)))


        if inversions:
            fname = "data/lists/chairs_list.txt"
            with open(fname, "r") as f:
                f1 = f.readlines()
            f1 = [f[:-1] for f in f1]

        if not train:
            fname = "data/lists/sv2_chairs_test.json"
            with open(fname, "r") as f:
                self.file_list = json.load(f)['ShapeNetV2']['03001627']
            print("Test List has : ", len(self.file_list), " files")
        else:
            fname = "data/lists/sv2_chairs_train.json"
            with open(fname, "r") as f:
                self.file_list = json.load(f)['ShapeNetV2']['03001627']
            fname = "data/lists/sv2_chairs_test.json"
            with open(fname, "r") as f:
                self.test_file_list = json.load(f)['ShapeNetV2']['03001627']
            self.file_list = self.file_list + [f for f in f1 if f not in self.test_file_list]
            print("Train List has : ", len(self.file_list), " files")

        if inversions:
            for f in self.file_list:
                if f not in f1:
                    self.file_list.remove(f)

        self.fname = fname

        tmp_data = []
        tmp_list = []
        for file in self.data:
            f = file.split('/')[-2]
            if f in self.file_list:
                tmp_list.append(f)
                tmp_data.append(file)
        tmp_list = set(tmp_list)
        for f in self.file_list:
            if f not in tmp_list:
                self.file_list.remove(f)

        # Filter inversion_ar to contain files in file_list
        f1 = [f for f in f1 if f in self.file_list]
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
        print("Collecting sketches from : ", self.root_dir)
        for files in os.listdir(self.root_dir):
            for file in os.listdir(os.path.join(self.root_dir, files)):
                if self.view_ != 'all':
                    # Single View
                    if file.endswith('.png') and int(file.split('_')[-1].split(".")[0]) in self.view_:
                        self.data.append(os.path.join(self.root_dir,files, file))
                else:
                    # All Views
                    if file.endswith('.png'):
                        self.data.append(os.path.join(self.root_dir,files, file))
                        


    def crop_from_params(self, input_imp,crop_params):
        min_x, min_y, max_x, max_y = crop_params
        min_x = int(min_x)
        min_y = int(min_y)
        max_x = int(max_x)
        max_y = int(max_y)

        img = input_imp
        
        cropped = img.crop((min_x, min_y, max_x, max_y))
        max_size = max(cropped.width, cropped.height)
        square_img = Image.new('RGB', (max_size, max_size), color='black')
        pad_x = (max_size - cropped.width) // 2
        pad_y = (max_size - cropped.height) // 2
        square_img.paste(cropped, (pad_x, pad_y))
        return square_img

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.preprocessor(img)
        view_ = img_path.split('/')[-1].split('_')[-1].split(".")[0]
        f_name = img_path.split('/')[-2]
        all_segments = []

        f_name_dir = os.path.join(self.root_dir, f_name)
        view_file = f"color_{view_}_crop_params.txt"
        
        with open(os.path.join(f_name_dir, view_file), "r") as f:
            crop_params = f.readline().split(",")

        for i in range(16):
            segment_map_path = os.path.join(self.segment_dir, f_name, str(i), f"color_{view_}.png")
            segment_map = Image.open(segment_map_path).convert('L')
            segment_map = self.crop_from_params(segment_map, crop_params).convert('RGB')
            segment_map = self.preprocessor(segment_map)[0]
            segment_map[segment_map > 0.5] = 1
            segment_map[segment_map <= 0.5] = 0
            all_segments.append(segment_map.unsqueeze(0))

        all_segments = torch.cat(all_segments, dim=0)
        return img, all_segments, f_name