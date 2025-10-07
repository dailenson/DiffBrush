import os
import random
from PIL import Image
import torch

from data_loader.base_dataset import BaseDataset, GenerateDataset

letters = " _!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
fixed_len = 1024

class IAMDataset(BaseDataset):
    def __init__(self, image_path, style_path, text_path, type, content_type='unifont'):
        configs = {'image_path':image_path, 'style_path':style_path, 'type':type, 'content_type':content_type, 
                   'fixed_len':fixed_len, 'text_path':text_path, 'letters':letters,}
        super(IAMDataset, self).__init__(configs)


class IAMGenerateDataset(GenerateDataset):
    def __init__(self, style_path, type, ref_num, content_type='unifont'):
        configs = {'style_path':style_path, 'type':type, 'content_type':content_type, 
                   'fixed_len':fixed_len, 'letters':letters, 'ref_num':ref_num}
        super(IAMGenerateDataset, self).__init__(configs)
