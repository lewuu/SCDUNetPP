import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SwinT_scdunetpp import SwinTransformer
# from SwinT_scdunetpp import SwinTransformer


class BasicConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 act_cfg='GELU'):
        super(BasicConvBlock, self).__init__()

        convs = []
        for i in range(num_convs):
            convs.extend([
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size if i == 0 else 3,
                    stride=stride if i == 0 else 1,
                    padding=0 if kernel_size==1 else dilation,
                    dilation=1 if kernel_size==1 else dilation, 
                    bias=False, padding_mode='reflect'),
                nn.BatchNorm2d(out_channels),
                nn.GELU() if act_cfg=='GELU' else nn.LeakyReLU()
            ])

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


class SpatialPooling(nn.Sequential):
    def __init__(self, in_chans, out_chans):
        super(SpatialPooling, self).__init__()
        self.apool_h = nn.AdaptiveAvgPool2d((None, 1)) 
        self.apool_w = nn.AdaptiveAvgPool2d((1, None))
        self.mpool_h = nn.AdaptiveMaxPool2d((None, 1)) 
        self.mpool_w = nn.AdaptiveMaxPool2d((1, None))
        self.conv11 = BasicConvBlock(in_chans*2,out_chans,1,1)

    def forward(self, x):
        x_ah = self.apool_h(x)
        x_aw = self.apool_w(x)
        x_mh = self.mpool_h(x)
        x_mw = self.mpool_w(x)
        x_a = torch.matmul(x_ah, x_aw)
        x_m = torch.matmul(x_mh, x_mw)
        x = self.conv11(torch.cat([x_a,x_m],1))
        return x


class SpectralPooling(nn.Sequential):
    def __init__(self, in_chans, out_chans):
        super(SpectralPooling, self).__init__()
        self.apool_c = nn.AdaptiveAvgPool2d(1)
        self.mpool_c = nn.AdaptiveMaxPool2d(1)
        self.conv11 = BasicConvBlock(in_chans*2,out_chans,1,1)

    def forward(self, x):
        n,c,h,w = x.shape
        x_c = [self.apool_c(x),self.mpool_c(x)]
        x = self.conv11(torch.cat(x_c,1))
        co = x.shape[1]
        return x.expand([n,co,h,w])


class DetailedSSAggregationBlock(nn.Module):
    def __init__(self, in_chans, out_chans=256, stride=2):
        super(DetailedSSAggregationBlock, self).__init__()
        self.in_chans= 15 if in_chans ==21 else in_chans

        self.conv1 = BasicConvBlock(self.in_chans,out_chans,1,3)
        self.conv2 = BasicConvBlock(2*out_chans,out_chans,1,1)

        modules = []
        for rate in (1,2,3):
            modules.append(BasicConvBlock(out_chans,out_chans,1,3,dilation=rate))

        modules.append(SpatialPooling(out_chans, out_chans))
        modules.append(SpectralPooling(out_chans, out_chans))

        self.mods = nn.ModuleList(modules)

        self.project = nn.Sequential(
            BasicConvBlock((len(self.mods)) * out_chans, out_chans,1,1,1),
            nn.Dropout(0.5))

    def forward(self, x):
        x1 = self.conv1(x[0][:,0:self.in_chans,:,:])
        x2 = self.conv2(torch.cat([x1,x[1]],dim=1))
        res = []
        for mod in self.mods:
            res.append(mod(x2))
        return self.project(torch.cat(res, dim=1))


class UpSampling(nn.Module):
    def __init__(self, scale_factor=None, mode='nearest', align_corners=False):
        super(UpSampling, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if self.mode == 'nearest':
            out = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        elif self.mode == 'bilinear':
            out = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,align_corners=self.align_corners)
        else:
            raise ValueError("Not implemented upsampling mode.")
        return out


class DecoderBlock(nn.Module):
    def __init__(self,
                 up_channels=64,
                 cat_channels=32,
                 n_cat=0,
                 scale_factor=2
                 ):
        super().__init__()

        in_chans = cat_channels*n_cat+up_channels
        out_chans = cat_channels
        self.upsize = UpSampling(scale_factor,mode='nearest',align_corners=False)
        self.double_conv = BasicConvBlock(in_chans,out_chans,2)

    def forward(self, x):
        x[-1] = self.upsize(x[-1])
        x = torch.cat(x, dim=1)
        x = self.double_conv(x)

        return x


class SCDUNetPP(nn.Module): 
    def __init__(self,
                 in_chans=3,
                 num_class=2
                 ):
        super().__init__()

        self.wt = nn.Parameter(torch.FloatTensor(5))
        self.wt.data.fill_(1)

        self.swin = SwinTransformer(img_size=128,in_chans=in_chans,embed_dim=64, depths=[2, 6, 2], num_heads=[2, 4, 8],out_indices=(0, 1, 2),window_size=8) 

        n1 = 32
        base = in_chans*2 if in_chans==21 else n1
        filters = [base, n1*2, n1*2, n1*4, n1*8]
        
        self.dssa = DetailedSSAggregationBlock(in_chans, filters[0])
        
        self.firstconv = BasicConvBlock(in_chans,filters[0],2)

        # Stem
        self.stem = BasicConvBlock(in_chans,filters[1],3,stride=2)

        # Decoder
        self.decoder0_1 = DecoderBlock(filters[1], filters[0], 1)
        self.decoder1_1 = DecoderBlock(filters[2], filters[1], 1)
        self.decoder2_1 = DecoderBlock(filters[3], filters[2], 1)
        self.decoder3_1 = DecoderBlock(filters[4], filters[3], 1)

        self.decoder0_2 = DecoderBlock(filters[1], filters[0], 2)
        self.decoder1_2 = DecoderBlock(filters[2], filters[1], 2)
        self.decoder2_2 = DecoderBlock(filters[3], filters[2], 2)

        self.decoder0_3 = DecoderBlock(filters[1], filters[0], 3)
        self.decoder1_3 = DecoderBlock(filters[2], filters[1], 3)

        self.decoder0_4 = DecoderBlock(filters[1], filters[0], 4)

        self.logit = nn.Conv2d(filters[0], num_class, kernel_size=1)
 
    def forward_features(self, x):

        x0_0 = self.firstconv(x) 
        x1_0 = self.stem(x)
        x2_0,x3_0,x4_0 = self.swin(x) 

        x0_1 = self.decoder0_1([x0_0, x1_0])  

        x_d = self.dssa([x,x0_1])

        x1_1 = self.decoder1_1([x1_0, x2_0])
        x0_2 = self.decoder0_2([x0_0, x0_1, x1_1])

        x2_1 = self.decoder2_1([x2_0, x3_0])
        x1_2 = self.decoder1_2([x1_0, x1_1, x2_1])
        x0_3 = self.decoder0_3([x0_0, x0_1, x0_2, x1_2])

        x3_1 = self.decoder3_1([x3_0, x4_0])
        x2_2 = self.decoder2_2([x2_0, x2_1, x3_1])
        x1_3 = self.decoder1_3([x1_0, x1_1, x1_2, x2_2])
        x0_4 = self.decoder0_4([x0_0, x0_1, x0_2, x0_3, x1_3])

        features = [x0_1,x0_2,x0_3,x0_4,x_d]

        return tuple(features)

    def forward(self,x):
        H,W = x.shape[2:]
        x = self.forward_features(x)
        logits=[]
        for i, fea in enumerate(x):
            logits.append(self.wt[i]*self.logit(fea))
        logit = torch.sum(torch.stack(logits),dim=0)

        if H!=logit.shape[2] or W!=logit.shape[3]:
            print('The input-output sizes do not match.')
            logit = F.interpolate(logit, size=(H,W), mode='bilinear', align_corners=False)

        return logit

    def freeze_param(self):
        blks = [self.dssa, self.firstconv, self.stem, self.swin]
        for blk in blks:
            blk.eval()
            for param in blk.parameters():
                param.requires_grad = False


if __name__ == '__main__':

    x=torch.randn(8,21,128,128) 
    model=SCDUNetPP(in_chans=21, num_class=2)
    print(model(x).shape) 
