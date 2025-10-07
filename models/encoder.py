import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
from models.transformer import *
from einops import rearrange, repeat
from models.loss import Proxy_Anchor
from models.resnet_dilation import resnet18 as resnet18_dilation
import torch.fft as fft

### merge the handwriting style and printed content
class Mix_TR(nn.Module):
    def __init__(self, nb_classes, d_model=256, nhead=8, num_encoder_layers=2, num_head_layers=1, num_decoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation="relu", return_intermediate_dec=False,
                 normalize_before=True, fft_threshold=8):
        super(Mix_TR, self).__init__()
        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.base_encoder = TransformerEncoder(encoder_layer, num_encoder_layers, None)

        vertical_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.vertical_head = TransformerEncoder(encoder_layer, num_head_layers, vertical_norm)

        horizontal_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.horizontal_head = TransformerEncoder(encoder_layer, num_head_layers, horizontal_norm)

        ### fusion the content and style in the transformer decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        vertical_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.vertical_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, vertical_decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        
        horizontal_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.horizontal_decoder = TransformerDecoder(decoder_layer, num_decoder_layers, horizontal_decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        
        self.add_position1D = PositionalEncoding(dropout=0.1, dim=d_model) # add 1D position encoding
        self.add_position2D = PositionalEncoding2D(dropout=0.1, d_model=d_model) # add 2D position encoding

        self.vertical_pro_mlp = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
        self.horizontal_pro_mlp = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
        self._reset_parameters()

        ### style encoder
        self.Feat_Encoder = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.Feat_Encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.Feat_Encoder.layer4 = nn.Identity()
        self.Feat_Encoder.fc = nn.Identity()
        self.Feat_Encoder.avgpool = nn.Identity()
        self.style_dilation_layer = resnet18_dilation().conv5_x
        ### content encoder
        self.content_encoder = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(models.resnet18(weights='ResNet18_Weights.DEFAULT').children())[1:-2]))

        # proxy projection head
        self.vertical_proxy = Proxy_Anchor(nb_classes=nb_classes, sz_embed=d_model, proxy_mode='only_real')
        self.horizontal_proxy = Proxy_Anchor(nb_classes=nb_classes, sz_embed=d_model, proxy_mode='only_real')

        ### fft
        self.fft_threshold = fft_threshold

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def random_double_sampling(self, x, ratio=0.25):
        """
        Sample the positive pair (i.e., o and o^+) within a character by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [L, B, N, D], sequence
        return o [B, N, 1, D], o^+ [B, N, 1, D]
        """
        L, B, N, D = x.shape  # length, batch, group_number, dim
        x = rearrange(x, "L B N D -> B N L D")
        noise = torch.rand(B, N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=2)

        anchor_tokens, pos_tokens = int(L*ratio), int(L*2*ratio)
        ids_keep_anchor, ids_keep_pos = ids_shuffle[:, :, :anchor_tokens], ids_shuffle[:, :, anchor_tokens:pos_tokens]
        x_anchor = torch.gather(
            x, dim=2, index=ids_keep_anchor.unsqueeze(-1).repeat(1, 1, 1, D))
        x_pos = torch.gather(
            x, dim=2, index=ids_keep_pos.unsqueeze(-1).repeat(1, 1, 1, D))
        return x_anchor, x_pos
    

    def random_vertical_sample(self, x, ratio=0.5):
        x = rearrange(x, "( H W ) B D -> B H W D", H=4)
        B, H, W, D = x.shape
        noise = torch.rand(B, H, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)
        tokens = int(H*ratio)
        x_sample = torch.gather(
            x, dim=1, index=ids_shuffle[:, :tokens].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, W, D))
        x_sample = rearrange(x_sample, 'B H W D -> ( H W ) B D')
        
        return x_sample
    

    def random_horizontal_sample(self, x, ratio=0.5):
        x = rearrange(x, "( H W ) B D -> B W H D", H=4)
        B, W, H, D = x.shape
        noise = torch.rand(B, W, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)
        tokens = int(W*ratio)
        x_sample = torch.gather(
            x, dim=1, index=ids_shuffle[:, :tokens].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, D))
        x_sample = rearrange(x_sample, 'B W H D -> ( H W ) B D')
        
        return x_sample

    def forward(self, style, content, wid):
        batch_size, in_planes, h, w = style.shape
        # CNN + Transformer encode
        style = self.Feat_Encoder(style)
        style = rearrange(style, 'n (c h w) ->n c h w', c=256, h=4).contiguous()
        style = self.style_dilation_layer(style)    # [B, 512, 4, W]  W ≤ 64
        style = self.add_position2D(style)
        style = rearrange(style, 'n c h w ->(h w) n c').contiguous()
        base_style = self.base_encoder(style)   # [4*W, B, 512]

        vertical_style = self.vertical_head(base_style) # [4*W, B, 512]
        horizontal_style = self.horizontal_head(base_style)

        # 横向采样, 使模型关注垂直风格信息(每个字母所处高度)
        vertical_style_proxy = self.random_horizontal_sample(vertical_style)    # # [2*W, B, 512]
        vertical_style_proxy = self.vertical_pro_mlp(vertical_style_proxy)
        vertical_style_proxy = torch.mean(vertical_style_proxy, dim=0)
        vertical_style_loss = self.vertical_proxy(vertical_style_proxy, wid)

        # 纵向采样, 使模型关注水平风格信息(字符间距, 字母间的连笔, 单词之间的行间距)
        horizontal_style_proxy = self.random_vertical_sample(horizontal_style)
        horizontal_style_proxy = self.horizontal_pro_mlp(horizontal_style_proxy)
        horizontal_style_proxy = torch.mean(horizontal_style_proxy, dim=0)
        horizontal_style_loss = self.horizontal_proxy(horizontal_style_proxy, wid)

        # content encoder
        content = rearrange(content, 'n t h w ->(n t) 1 h w').contiguous()
        content = self.content_encoder(content)
        content = rearrange(content, '(n t) c h w ->t n (c h w)', n=batch_size).contiguous() # n is batch size
        content = self.add_position1D(content)
        
        style_hs = self.horizontal_decoder(content, horizontal_style, tgt_mask=None)
        hs = self.vertical_decoder(style_hs[0], vertical_style, tgt_mask=None)
        
        return hs[0].permute(1, 0, 2).contiguous(), vertical_style_loss, horizontal_style_loss # n t c
    
    def generate(self, style, content):
        batch_size, in_planes, h, w = style.shape
        # CNN + Transformer encode
        style = self.Feat_Encoder(style)
        style = rearrange(style, 'n (c h w) ->n c h w', c=256, h=4).contiguous()
        style = self.style_dilation_layer(style)    # [B, 512, 4, W]
        style = self.add_position2D(style)
        style = rearrange(style, 'n c h w ->(h w) n c').contiguous()
        base_style = self.base_encoder(style)   # [4*W, B, 512]

        vertical_style = self.vertical_head(base_style) # [4*W, B, 512]
        horizontal_style = self.horizontal_head(base_style)

        # 横向上打乱(列打乱), 使模型关注垂直风格信息(每个字母所处高度)
        # horizontal_style_proxy = self.random_horizontal_shuffle(vertical_style, blocks=64)
        # horizontal_style_proxy = rearrange(horizontal_style_proxy, 'h w n c->(h w) n c').contiguous()

        # 纵向上打乱(行打乱), 使模型关注水平风格信息(字符间距, 字母间的连笔, 单词之间的行间距)
        # horizontal_style = self.random_vertical_shuffle(horizontal_style)
        # horizontal_style = rearrange(horizontal_style, 'h w n c->(h w) n c').contiguous()

        # content encoder
        content = rearrange(content, 'n t h w ->(n t) 1 h w').contiguous()
        content = self.content_encoder(content)
        content = rearrange(content, '(n t) c h w ->t n (c h w)', n=batch_size).contiguous() # n is batch size
        content = self.add_position1D(content)
        
        style_hs = self.horizontal_decoder(content, horizontal_style, tgt_mask=None)
        hs = self.vertical_decoder(style_hs[0], vertical_style, tgt_mask=None)
        
        return hs[0].permute(1, 0, 2).contiguous()

