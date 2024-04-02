"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import os.path
import random
import torch


class TemplateDataset(BaseDataset):
    '''
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser
    '''
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.dir_M = os.path.join(opt.dataroot, opt.phase + 'Mask')
        self.modal = ['flair', 't1', 't1ce', 't2']
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 4))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 4))

        self.transform_M = get_transform(self.opt, grayscale=1)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        btoa = self.opt.test_direction == 'BtoA'
        A = []
        B = []
        M = []
        if btoa:
            AM_path = []
            BM_path = []
            B_path = self.B_paths[index % self.B_size]  # make sure index is within then range
            B_name = B_path.split('/')[-1]
            slice_n = B_path.split('/')[-2]
            A_paths = make_dataset(os.path.join(self.dir_A,slice_n),self.opt.max_dataset_size)
            A_path = A_paths[random.randint(0, len(A_paths) - 1)]
            A_name = A_path.split('/')[-1]
            slice_n = A_path.split('/')[-2]
            nameA_path =  os.path.dirname(os.path.dirname(A_path))
            for m_name in self.modal:                
                AM_path.append(os.path.join(nameA_path,m_name,A_name))

            M_path = os.path.join(self.dir_M,A_name)
            nameB_path = os.path.dirname(os.path.dirname(B_path))
            for m_name in self.modal:                
                BM_path.append(os.path.join(nameB_path,m_name,B_name))  
        else:
            AM_path = []
            BM_path = []
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            A_name = A_path.split('/')[-1]
            slice_n = A_path.split('/')[-2]
            nameA_path =  os.path.dirname(os.path.dirname(A_path))
            for m_name in self.modal:                
                AM_path.append(os.path.join(nameA_path,m_name,A_name))

            M_path = os.path.join(self.dir_M,A_name)

            B_paths = make_dataset(os.path.join(self.dir_B,slice_n),self.opt.max_dataset_size)
            B_path = B_paths[random.randint(0, len(B_paths) - 1)]
            B_name = B_path.split('/')[-1]
            nameB_path = os.path.dirname(os.path.dirname(B_path))
            for m_name in self.modal:                
                BM_path.append(os.path.join(nameB_path,m_name,B_name))    

        for A_path_num in AM_path:        
            A_img = Image.open(A_path_num)#.convert('L')
            A.append(self.transform_A(A_img))

        for B_path_num in BM_path:
            B_img = Image.open(B_path_num)#.convert('L')
            B.append(self.transform_B(B_img))

        M_img = Image.open(M_path)#.convert('L')
        # apply image transformation
        M = self.transform_M(M_img)
        # M = M.unsqueeze(0)
        # M = M.repeat(4,1,1)
        A = torch.cat(A,0)
        B = torch.cat(B,0)

        return {'A': A, 'B': B, 'M': M, 'A_paths': A_path, 'B_paths': B_path, 'M_paths': M_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        btoa = self.opt.test_direction == 'BtoA'
        if btoa:
            print(self.B_size)
            return self.B_size
        else:
            return self.A_size