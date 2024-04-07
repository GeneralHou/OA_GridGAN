import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        
        dir_A = '_A'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        
        
        dir_H = '_H'
        self.dir_H = os.path.join(opt.dataroot, opt.phase + dir_H)
        self.H_paths = sorted(make_dataset(self.dir_H))


        
        if opt.isTrain:
            if self.opt.label_nc == 0: dir_B = '_B'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        
        A_path = self.A_paths[index]              
        A = Image.open(A_path).convert('RGB')
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A)

        
        
        H_path = self.H_paths[index]
        H = Image.open(H_path).convert('RGB')
        params = get_params(self.opt, H.size)
        transform_H = get_transform(self.opt, params)
        H_tensor = transform_H(H)

        B_tensor = 0
        
        if self.opt.isTrain:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        input_dict = {'label': A_tensor, 'label_h': H_tensor, 'image': B_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'