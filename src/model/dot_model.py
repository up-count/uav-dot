import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationHead


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
            

    def forward(self, x):
        identity = x
    
        _, _, h, w = x.size()
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm,
                      attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        # remove first skip with same spatial resolution
        features = features[1:]
        # reverse channels to start from head of encoder
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        out = []
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            out.append(x)

        return out[::-1]

class PixelDistillBlock(nn.Module):
    def __init__(self, in_channels, features_channels, scale_down=1):
        super(PixelDistillBlock, self).__init__()
        
        if scale_down > 1:
            self.down = nn.PixelUnshuffle(scale_down)
            in_channels = int(in_channels * scale_down**2)
        else:
            self.down = nn.Identity()
        
        self.conv = nn.Conv2d(in_channels, features_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(features_channels)
        self.relu = nn.ReLU()
        
        self.attn = CoordAtt(inp=features_channels, oup=features_channels)
        
    def forward(self, x):
        x = self.down(x)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.attn(x)
        
        return x

class PixelDistill(nn.Module):
    def __init__(self, in_channels, out_channels, features_channels, down_scale=1):
        super(PixelDistill, self).__init__()
        self.down_reduction = int(2**2)
        
        self.unshuffle = nn.PixelUnshuffle(2)
                
        self.blocks = nn.ModuleList([
            PixelDistillBlock(in_channels, features_channels//2, scale_down=down_scale//2) for _ in range(self.down_reduction)
        ])
        
        self.conv_1 = nn.Conv2d(int(self.down_reduction*(features_channels//2)), features_channels, 3, 1, 1)
        self.bn_1 = nn.BatchNorm2d(features_channels)
        self.relu = nn.ReLU()
        
        self.conv_2 = nn.Conv2d(features_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.unshuffle(x)
        
        outs = []
        for i, block in enumerate(self.blocks):
            outs.append(block(x[:, i::self.down_reduction]))
        
        x = torch.cat(outs, dim=1)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        
        x = self.conv_2(x)
        
        return x

class DotModel(torch.nn.Module):
    def __init__(self, 
                 encoder_name, 
                 classes, 
                 image_size: tuple[int, int],
                 mask_size: tuple[int, int],
                 reduce_spatial_mode='interpolate') -> None:
        super().__init__()

        encoder_weights = 'imagenet'
        encoder_depth = 5
        in_channels = 3
        decoder_use_batchnorm: bool = True
        decoder_channels = (256, 128, 64, 32, 16)
        decoder_attention_type = None
        classes = classes
        down_scale = int(image_size[0]/mask_size[0])

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = nn.ModuleList()
        for channel in decoder_channels[::-1]:
            self.segmentation_head.append(
                SegmentationHead(
                    in_channels=channel,
                    out_channels=classes,
                    activation=None,
                    kernel_size=3,
                )
            )

        if reduce_spatial_mode == 'interpolate':
            self.pre = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=1./down_scale, mode='bilinear', align_corners=True),
            )

        elif reduce_spatial_mode == 'pixel':
            self.pre = PixelDistill(in_channels, in_channels, features_channels=32, down_scale=down_scale)
        elif reduce_spatial_mode == 'none':
            self.pre = torch.nn.Identity()
        else:
            raise ValueError('Unknown reduce_spatial_mode')

    def forward(self, x):
        x = self.pre(x)
        
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = []
        for i, seg_head in enumerate(self.segmentation_head):
            masks.append(seg_head(decoder_output[i]))

        return masks[:3]

if __name__ == '__main__':
    
    model = DotModel('mit_b2', 1, (1920, 1088), (960, 544), 'pixel')
    
    x = torch.randn(1, 3, 1088, 1920)
    
    y = model(x)
    print(y[0].shape)
