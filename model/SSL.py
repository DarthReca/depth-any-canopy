import lightning as pl
import torch
from torch import nn

from .backbone import SSLVisionTransformer
from .dpt_head import DPTHead


class SSLAE(nn.Module):
    def __init__(self, pretrained=None, classify=True, n_bins=256, huge=False):
        super().__init__()
        if huge == True:
            self.backbone = SSLVisionTransformer(
                embed_dim=1280,
                num_heads=20,
                out_indices=(9, 16, 22, 29),
                depth=32,
                pretrained=pretrained,
            )
            self.decode_head = DPTHead(
                classify=classify,
                in_channels=(1280, 1280, 1280, 1280),
                embed_dims=1280,
                post_process_channels=[160, 320, 640, 1280],
            )
        else:
            self.backbone = SSLVisionTransformer(pretrained=pretrained)
            self.decode_head = DPTHead(classify=classify, n_bins=256)

    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x)
        return x


class SSLModule(pl.LightningModule):
    def __init__(self, ssl_path="compressed_SSLbaseline.pth"):
        super().__init__()

        if "huge" in ssl_path:
            self.chm_module_ = SSLAE(classify=True, huge=True).eval()
        else:
            self.chm_module_ = SSLAE(classify=True, huge=False).eval()

        if "compressed" in ssl_path:
            ckpt = torch.load(ssl_path, map_location="cpu")
            self.chm_module_ = torch.quantization.quantize_dynamic(
                self.chm_module_,
                {torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d},
                dtype=torch.qint8,
            )
            self.chm_module_.load_state_dict(ckpt, strict=False)
        else:
            ckpt = torch.load(ssl_path)
            state_dict = ckpt["state_dict"]
            self.chm_module_.load_state_dict(state_dict)

        self.chm_module = lambda x: 10 * self.chm_module_(x)

    def forward(self, x):
        x = self.chm_module(x)
        return x
