import os
import argparse
import random
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.IAMDataset import IAMGenerateDataset
from models.unet import UNetModel
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
import torch.distributed as dist
from tqdm import tqdm
from utils.util import fix_seed
import numpy as np
from PIL import Image

WRITER_NUMS = 496

def main(args):
    """ load config file into cfg"""
    cfg_from_file(args.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)
    
    fid_dir = os.path.join(args.save_dir, 'fid')
    fid_wri_dir = os.path.join(args.save_dir, 'fid_wri')
    hwd_dir = os.path.join(args.save_dir, 'hwd')
    os.makedirs(fid_dir, exist_ok=True)
    os.makedirs(fid_wri_dir, exist_ok=True)
    os.makedirs(hwd_dir, exist_ok=True)

    """ set mulit-gpu """
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    totol_process = dist.get_world_size()

    """build model architecture"""
    diffusion = Diffusion(device=args.device)
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM, nb_classes=WRITER_NUMS).to(args.device)
    
    """load pretrained model"""
    if len(args.pretrained_model) > 0: 
        unet.load_state_dict(torch.load(f'{args.pretrained_model}', map_location=torch.device('cpu')))
        print('load pretrained model from {}'.format(args.pretrained_model))
    else:
        raise IOError('input the correct checkpoint path')
    unet.eval()

    """Load and Freeze VAE Encoder"""
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
    vae = vae.to(args.device)
    vae.requires_grad_(False)

    """Generate and FID-related and HWD"""
    text_corpus = 'data/wikitext103.te'
    with open(text_corpus, 'r') as _f:
        texts = _f.read().splitlines()
    each_process = len(texts)//totol_process
    if  len(texts)%totol_process == 0:
        temp_texts = texts[dist.get_rank()*each_process:(dist.get_rank()+1)*each_process]
    else:
        each_process += 1
        temp_texts = texts[dist.get_rank()*each_process:(dist.get_rank()+1)*each_process]

    if 'IAM' in args.cfg_file:
        dataset_name = 'IAM'
        dataset = IAMGenerateDataset(cfg.TEST.STYLE_PATH,'test', len(temp_texts))
    elif 'RIMES' in args.cfg_file:
        dataset_name = 'RIMES'
        dataset = None
    else:
        raise IOError('input the correct dataset name')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.DATA_LOADER.NUM_THREADS, pin_memory=True)
    
    writer_num_each_time = 65
    with torch.no_grad():
        loader_iter = iter(data_loader)
        for x_text in tqdm(temp_texts, position=0, desc='batch_number'):
            data = next(loader_iter)
            style_ref, wid = data['style'][0], data['wid']
            style_idx = data['style_idx']
            
            loader = []
            for i in range(len(style_ref) // writer_num_each_time + 1):
                loader.append((style_ref[i * writer_num_each_time:(i + 1) * writer_num_each_time], 
                               wid[i * writer_num_each_time:(i + 1) * writer_num_each_time], 
                               style_idx[i * writer_num_each_time:(i + 1) * writer_num_each_time]))
            
            for (style_ref, wid, style_idx) in loader:
                style_input = style_ref.to(args.device)
                text_ref = dataset.get_content(x_text)
                text_ref = text_ref.to(args.device).repeat(style_input.shape[0], 1, 1, 1)
                x = torch.randn((text_ref.shape[0], 4, style_ref.shape[2]//8, (dataset.fixed_len)//8)).to(args.device)
                ema_sampled_images = diffusion.ddim_sample(unet, vae, style_input.shape[0], 
                                                        x, style_input, text_ref,
                                                        args.sampling_timesteps, args.eta)
                for index in range(len(ema_sampled_images)):
                    im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
                    image = im.convert("L")
                    os.makedirs(os.path.join(fid_wri_dir, wid[index][0]), exist_ok=True)
                    image.save(os.path.join(fid_wri_dir, wid[index][0], f"{wid[index][0]}-{x_text}_{style_idx[index][0]}.png"))
                    image.save(os.path.join(hwd_dir, f"{wid[index][0]}-{x_text}_{style_idx[index][0]}.png"))
                    
    """Generate and Calculate FID"""
    text_corpus = 'data/wikitext103.te'
    with open(text_corpus, 'r') as _f:
        texts = _f.read().splitlines()
    temp_texts = []
    for text in texts:
        if 35 <= len(text) <= 61:
            temp_texts.append(text)
    texts = temp_texts
    each_process = len(texts)//totol_process
    if  len(texts)%totol_process == 0:
        temp_texts = texts[dist.get_rank()*each_process:(dist.get_rank()+1)*each_process]
    else:
        each_process += 1
        temp_texts = texts[dist.get_rank()*each_process:(dist.get_rank()+1)*each_process]
    if 'IAM' in args.cfg_file:
        dataset_name = 'IAM'
        dataset = IAMGenerateDataset(cfg.TEST.STYLE_PATH, 'test', len(texts))
    elif 'RIMES' in args.cfg_file:
        dataset_name = 'RIMES'
        dataset = None
    else:
        raise IOError('input the correct dataset name')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.DATA_LOADER.NUM_THREADS, pin_memory=True)

    writer_num_each_time = 65
    with torch.no_grad():
        loader_iter = iter(data_loader)
        for _ in tqdm(range(dataset.__len__())):
            x_text = random.choice(temp_texts)
            data = next(loader_iter)
            style_ref,wid = data['style'][0],data['wid']
            style_idx = data['style_idx']
            
            loader = []
            for i in range(len(style_ref) // writer_num_each_time + 1):
                loader.append((style_ref[i * writer_num_each_time:(i + 1) * writer_num_each_time], 
                               wid[i * writer_num_each_time:(i + 1) * writer_num_each_time], 
                               style_idx[i * writer_num_each_time:(i + 1) * writer_num_each_time]))
                
            for (style_ref,wid, style_idx) in loader:
                style_input = style_ref.to(args.device)
                text_ref = dataset.get_content(x_text)
                text_ref = text_ref.to(args.device).repeat(style_input.shape[0], 1, 1, 1)
                x = torch.randn((text_ref.shape[0], 4, style_ref.shape[2]//8, (dataset.fixed_len)//8)).to(args.device)
                ema_sampled_images = diffusion.ddim_sample(unet, vae, style_input.shape[0], 
                                                        x, style_input, text_ref,
                                                        args.sampling_timesteps, args.eta)
                for index in range(len(ema_sampled_images)):
                    im = torchvision.transforms.ToPILImage()(ema_sampled_images[index])
                    image = im.convert("L")

                    if len(os.listdir(fid_dir)) >= 5000:
                        return
                    image.save(os.path.join(fid_dir, f"{wid[index][0]}-{x_text}_{style_idx[index][0]}.png"))
                    

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', default='configs/IAM.yml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--dir', dest='save_dir', default='Generated/x_start/99-epoch', help='target dir for storing the generated characters')
    parser.add_argument('--pretrained_model', dest='pretrained_model', default='', required=True, help='continue train model')
    parser.add_argument('--generate_type', dest='generate_type', default='oov_u', help='choose the setting of generated handwritten text')
    parser.add_argument('--device', type=str, default='cuda', help='device for test')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sampling_timesteps', type=int, default=50)
    parser.add_argument('--sample_method', type=str, default='ddim', help='choose the method for sampling')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--local_rank', type=int, default=0, help='device for training')
    args = parser.parse_args()
    main(args)
