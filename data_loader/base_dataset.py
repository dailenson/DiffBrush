import random
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
from torchvision import transforms
from PIL import Image
import torchvision
import cv2
import math

critical_width = 512

class BaseDataset(Dataset):
    def __init__(self, configs: dict):
        image_path, style_path= configs['image_path'], configs['style_path']
        type, content_type = configs['type'], configs['content_type']
        text_path, fixed_len, letters = configs['text_path'], configs['fixed_len'], configs['letters']
        
        self.fixed_len = fixed_len
        self.letters = letters
        self.data_dict = self.load_data(text_path)
        self.type = type
        self.image_path = os.path.join(image_path)
        self.style_path = os.path.join(style_path)
        self.tokens = {"PAD_TOKEN": len(letters)}
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.indices = list(self.data_dict.keys())
        self.transforms = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
        self.con_symbols = self.get_symbols(content_type)
        list_token = ['[GO]', '[END]', '[PAD]']
        self.character = list_token + list(letters)
        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i


    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            train_data = f.readlines()
            train_data = [i.strip() for i in train_data]
            full_dict = {}
            idx = 0
            for i in train_data:
                writer_id = i.split(' ',1)[0].split(',')[0]
                img_idx = i.split(' ',1)[0].split(',')[1]
                image = img_idx + '.png'
                transcription = i.split(' ',1)[1]
                full_dict[idx] = {'image': image, 'wid': writer_id, 'label':transcription}
                idx += 1
        return full_dict


    def load_short_data(self, data_path):
        with open(data_path, 'r') as f:
            train_data = f.readlines()
            train_data = [i.strip() for i in train_data]
            full_dict = {}
            for i in train_data:
                writer_id = i.split(' ',1)[0].split(',')[0]
                idx = i.split(' ',1)[0].split(',')[1]
                transcr = i.split(' ',1)[1]
                short_idx = idx.split('-')[0] + '-' + idx.split('-')[1]
                if full_dict.get(short_idx) is None:
                    full_dict[short_idx] = []
                full_dict[short_idx].append({'image': idx + '.png', 'wid': writer_id, 'label':transcr})
        
        return full_dict


    def get_style_ref(self, wr_id):
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        style_index = random.sample(range(len(style_list)), 1) # anchor and positive
        style_images = [cv2.imread(os.path.join(self.style_path, wr_id, style_list[index]), flags=0)
                        for index in style_index]
        height = style_images[0].shape[0]
        max_w = max([style_image.shape[1] for style_image in style_images])
        '''style images'''
        style_images = [style_image/255.0 for style_image in style_images]
        new_style_images = np.ones([1, height, max_w], dtype=np.float32)
        new_style_images[0, :, :style_images[0].shape[1]] = style_images[0]


    def get_symbols(self, input_type):
        with open(f"files/{input_type}.pickle", "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
        contents = []
        for char in self.letters:
            symbol = torch.from_numpy(symbols[ord(char)]).float()
            contents.append(symbol)
        contents.append(torch.zeros_like(contents[0])) # PAD_TOKEN image
        contents = torch.stack(contents)
        return contents


    def get_content(self, label):
        word_arch = [self.letter2index[i] for i in label]
        content_ref = self.con_symbols[word_arch]
        content_ref = 1.0 - content_ref
        return content_ref.unsqueeze(0)


    def concat_short_img(self, img, transcr):
        h, w, _ = img.shape
        space_w = w // len(transcr)
        repeat_width = space_w + w
        lower_bound = critical_width - w
        upper_bound = self.fixed_len - w
        ratio_lower = math.ceil(lower_bound / repeat_width)
        ratio_upper = upper_bound // repeat_width
        space = np.full((h, space_w, 3), 255, dtype=np.uint8)
        concat_transcr = ''
        if ratio_upper == 0:
            cat_img = img.copy()
            concat_transcr += transcr
        elif ratio_upper == 1:
            cat_img = cv2.hconcat([space, img.copy()])
            concat_transcr += ' ' + transcr
        else:
            if ratio_lower >= ratio_upper:
                times = random.randint(ratio_upper-1, ratio_upper)
            else:
                times = random.randint(ratio_lower, ratio_upper)
            tmp = cv2.hconcat([space, img.copy()])
            cat_img = tmp
            concat_transcr += ' ' + transcr
            for _ in range(1, times):
                cat_img = cv2.hconcat([cat_img, tmp])
                concat_transcr += ' ' + transcr
        img = cv2.hconcat([img, cat_img])
        if img.shape[1] > self.fixed_len:
            print(f'concated image width: { img.shape[1]}')
            raise ValueError('img.shape[1] > self.fixed_len')
        transcr += concat_transcr
        return img, transcr
        
    
    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['wid']
        transcr = label
        img_path = os.path.join(self.image_path, wr_id, image_name)
        image = Image.open(img_path).convert('RGB')
        # image.save('./image.png')
        if image.size[0] < critical_width and self.type == 'train':
            image = np.array(image)
            image, transcr = self.concat_short_img(image, transcr)
            image = Image.fromarray(image)
        image = self.transforms(image)
        
        style_ref= self.get_style_ref(wr_id)
        style_ref = torch.from_numpy(style_ref).to(torch.float32) # [1, h , w]

        contents = []
        for char in transcr:
            idx = self.letter2index[char]
            con_symbol = self.con_symbols[idx].numpy()
            thr = con_symbol==1.0
            prof = thr.sum(axis=0)
            on = np.argwhere(prof)[:,0]
            if len(on)>0:
                left = np.min(on)
                right = np.max(on)
                con_symbol = con_symbol[:,left-2:right+2]
            if len(on) == 0:
                con_symbol = con_symbol[:, 2:14]
            con_symbol = torch.from_numpy(con_symbol)
            contents.append(con_symbol)
        contents = torch.cat(contents, dim=-1)
        contents = contents.numpy()
        ratio_w = image.shape[-1]
        contents = cv2.resize(contents, (ratio_w, 64))
        contents = 1. - contents
        # cv2.imwrite('glyph.png', contents*255)
        contents = np.stack((contents, contents, contents), axis=2)
        glyph_line = self.transforms(contents)

        words = transcr.split(' ')
        # Get the position of the first and last characters of each word in the string
        transcr = str(transcr)
        h_ids, t_ids = [], []
        for word in words:
            if word == '':
                continue
            h_str = word[0]
            t_str = word[-1]
            h_idx = transcr.index(h_str, t_ids[-1] if t_ids else 0)
            t_idx = h_idx + len(word) - 1
            # h_idx = 0
            h_ids.append(h_idx)
            t_ids.append(t_idx)
        word_idx = [(h, t) for h, t in zip(h_ids, t_ids)]
        # Change all characters except a word into spaces
        word_transcrs = []
        for h, t in word_idx:
            word_transcr = transcr[h:t+1] + ' ' * len(transcr[t + 1:])
            word_transcrs.append(word_transcr)

        return {'img':image,
                'content':transcr, 
                'style':style_ref,
                'wid':int(wr_id),
                'transcr':transcr,
                'image_name':image_name,
                'word_idx': word_idx,
                'glyph_line': glyph_line, }


    def collate_fn_(self, batch):
        width = [item['img'].shape[2] for item in batch]
        c_width = [len(item['content']) for item in batch]
        s_width = [item['style'].shape[2] for item in batch]
        max_s_width = max(s_width)

        transcr = [item['transcr'] for item in batch]
        target_lengths = torch.IntTensor([len(t) for t in transcr])
        image_name = [item['image_name'] for item in batch]
        word_idx = [item['word_idx'] for item in batch]
        
        imgs = torch.full([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], self.fixed_len], fill_value=1., dtype=torch.float32)
        content_ref = torch.zeros([len(batch), max(c_width), 16 , 16], dtype=torch.float32)
        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width], dtype=torch.float32)
        target = torch.zeros([len(batch), max(target_lengths)], dtype=torch.int32)
        glyph_line = torch.ones([len(batch), batch[0]['glyph_line'].shape[0], batch[0]['glyph_line'].shape[1], self.fixed_len], dtype=torch.float32)

        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print('img', item['img'].shape)
            try:
                content = [self.letter2index[i] for i in item['content']]
                content = self.con_symbols[content]
                content_ref[idx, :len(content)] = content
            except:
                print('content', item['content'])
            target[idx, :len(transcr[idx])] = torch.Tensor([self.letter2index[t] for t in transcr[idx]])
            try:
                if max_s_width < self.fixed_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.fixed_len]
            except:
                print('style', item['style'].shape)
            glyph_line[idx, :, :, 0:item['glyph_line'].shape[2]] = item['glyph_line']
        
        lexicon, lexicon_length = self.encode(transcr)

        wid = torch.tensor([item['wid'] for item in batch])
        content_ref = 1.0 - content_ref # invert the image
        return {'img':imgs, 'style':style_ref, 'content':content_ref, 'wid':wid,
                'transcr': transcr, 'target':target, 'target_lengths':target_lengths, 'image_name':image_name, 'image_width': width,
                'lexicon': lexicon, 'lexicon_length':lexicon_length, 'glyph_line':glyph_line, 'word_idx': word_idx}

    def encode(self, text):
        """ convert a batch of text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) for s in text]
        batch_max_length  = max(length)+2
        # additional +2 for [GO] at the first step and [END] at the last step. batch_text is padded with [PAD] token after [END] token.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(self.dict['[PAD]'])
        batch_text[:, 0] = torch.LongTensor(len(text)).fill_(self.dict['[GO]'])
        for i, t in enumerate(text):
            text_new = list(t)
            text_new.append('[END]')
            text_new = [self.dict[char] if char in self.dict else len(self.dict) for char in text_new]
            batch_text[i][1:1 + len(text_new)] = torch.LongTensor(text_new)
        
        return (batch_text, torch.IntTensor(length))
    
    def decode(self, text_index):
        """ convert text-index into text-label. """
        text_index = text_index[:,1:]
        texts = []
        for index, t in enumerate(text_index):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            end_pos = text.find('[END]')
            text = text[:end_pos] if end_pos != -1 else text
            text = text.replace('[PAD]', 'P')
            texts.append(text)

        return texts
    
    def get_glyph(self, text, glyph_width):
        content = []
        for char in text:
            idx = self.letter2index[char]
            con_symbol = self.con_symbols[idx].numpy()
            thr = con_symbol==1.0
            prof = thr.sum(axis=0)
            on = np.argwhere(prof)[:,0]
            if len(on)>0:
                left = np.min(on)
                right = np.max(on)
                con_symbol = con_symbol[:,left-2:right+2]
            if len(on) == 0:
                con_symbol = con_symbol[:, 2:14]
            con_symbol = torch.from_numpy(con_symbol)
            content.append(con_symbol)
        content = torch.cat(content, dim=-1)
        content = content.numpy()
        content = cv2.resize(content, (glyph_width, 64))
        content = 1. - content
        content = np.stack((content, content, content), axis=2)
        content = self.transforms(content)
        glyph = torch.ones((3, 64, self.fixed_len))
        glyph[:,:, :content.shape[2]] = content

        return glyph

class GenerateDataset(Dataset):
    def __init__(self, configs: dict):
        style_path = configs['style_path']
        type, content_type = configs['type'], configs['content_type']
        fixed_len, letters = configs['fixed_len'], configs['letters']
        ref_num = configs['ref_num']
        
        self.fixed_len = fixed_len
        self.letters = letters
        self.style_path = os.path.join(style_path)
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.con_symbols = self.get_symbols(content_type)
        self.author_id = os.listdir(os.path.join(self.style_path))
        self.ref_num = ref_num
        
    def __len__(self):
        return self.ref_num


    def __getitem__(self, _):
        batch = []
        for wid in self.author_id:
            style_ref, style_idx = self.get_style_ref(wid)
            style_ref = torch.from_numpy(style_ref).unsqueeze(0)
            style_ref = style_ref.to(torch.float32)
            batch.append({'style':style_ref, 'wid':wid, 'style_idx':style_idx})
        
        s_width = [item['style'].shape[2] for item in batch]
        if max(s_width) < self.fixed_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.fixed_len
        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width], dtype=torch.float32)
        wid_list = []
        style_idx_list = []
        for idx, item in enumerate(batch):
            try:
                if max_s_width < self.fixed_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                else:
                    #new_style_image[:, :style_image.shape[1]] = style_image[:, :self.style_len]
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.fixed_len]
                wid_list.append(item['wid'])
                style_idx_list.append(item['style_idx'])
            except:
                print('style', item['style'].shape)
        
        return {'style':style_ref,'wid':wid_list, 'style_idx':style_idx_list}


    def get_style_ref(self, wr_id):
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        random.shuffle(style_list)
        for index in range(len(style_list)):
            style_ref = style_list[index]
            style_idx = style_ref.split('.')[0]
            style_image = cv2.imread(os.path.join(self.style_path, wr_id, style_ref), flags=0)
            if style_image.shape[1] > critical_width:
                break
            else:
                if index == len(style_list) - 1:
                    print(f'writer {wr_id} No style image found')
                continue
        style_image = style_image/255.0
        return style_image, style_idx


    def get_symbols(self, input_type):
        with open(f"files/{input_type}.pickle", "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
        contents = []
        for char in self.letters:
            symbol = torch.from_numpy(symbols[ord(char)]).float()
            contents.append(symbol)
        contents.append(torch.zeros_like(contents[0])) # PAD_TOKEN image
        contents = torch.stack(contents)
        return contents


    def get_content(self, label):
        word_arch = [self.letter2index[i] for i in label]
        content_ref = self.con_symbols[word_arch]
        content_ref = 1.0 - content_ref
        return content_ref.unsqueeze(0)
