#
# MC-Dropout U-Net
#
# Same depth / filter counts as networks/UNET.py (scores stay comparable to the
# baseline) with added nn.Dropout2d layers, kept ACTIVE at inference so we can
# draw T stochastic passes for uncertainty (Monte-Carlo Dropout, Gal &
# Ghahramani 2016; Bayesian SegNet, Kendall et al. 2015).
#
# Dropout is placed at the DEEP layers only, at 5 points:
#   - encoder: after contr_3 and contr_4  (perturbs the deep SKIP features + the
#              downstream path -- the skips are what bottleneck-only dropout could
#              not reach, which is why the epistemic signal was ~0 before)
#   - bottleneck: inside center
#   - decoder: after expand_4 and expand_3 (the two blocks nearest the bottleneck)
# The shallow high-res layers (contr_1/2, expand_1/2) are left clean to preserve
# boundary detail. Dropout has NO learnable parameters -> the weight count is
# identical to the baseline; only regularisation (train) and stochasticity
# (inference) change.
#
# Derived from the DKFZ basic_unet_example UNet. Apache License 2.0.
#

import torch
import torch.nn as nn


class MCDropoutUNet(nn.Module):

    def __init__(self, num_classes, in_channels=1, initial_filter_size=64,
                 kernel_size=3, dropout_p=0.4, do_instancenorm=True):
        super().__init__()
        self.dropout_p = dropout_p

        # MC dropout on the deep encoder skips + deep decoder blocks (the bottleneck
        # one lives inside self.center). These perturb the features that actually
        # reach the decoder, so the T MC samples differ -> real epistemic signal.
        self.drop_enc3 = nn.Dropout2d(p=dropout_p)
        self.drop_enc4 = nn.Dropout2d(p=dropout_p)
        self.drop_dec4 = nn.Dropout2d(p=dropout_p)
        self.drop_dec3 = nn.Dropout2d(p=dropout_p)

        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size*2, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)

        self.contr_3_1 = self.contract(initial_filter_size*2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size*2**2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)

        self.contr_4_1 = self.contract(initial_filter_size*2**2, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract(initial_filter_size*2**3, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)

        # bottleneck with MC dropout -- the ONLY change vs the baseline UNet
        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size*2**3, initial_filter_size*2**4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size*2**4, initial_filter_size*2**4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.ConvTranspose2d(initial_filter_size*2**4, initial_filter_size*2**3, 2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.expand_4_1 = self.expand(initial_filter_size*2**4, initial_filter_size*2**3)
        self.expand_4_2 = self.expand(initial_filter_size*2**3, initial_filter_size*2**3)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size*2**3, initial_filter_size*2**2, kernel_size=2, stride=2)

        self.expand_3_1 = self.expand(initial_filter_size*2**3, initial_filter_size*2**2)
        self.expand_3_2 = self.expand(initial_filter_size*2**2, initial_filter_size*2**2)
        self.upscale3 = nn.ConvTranspose2d(initial_filter_size*2**2, initial_filter_size*2, 2, stride=2)

        self.expand_2_1 = self.expand(initial_filter_size*2**2, initial_filter_size*2)
        self.expand_2_2 = self.expand(initial_filter_size*2, initial_filter_size*2)
        self.upscale2 = nn.ConvTranspose2d(initial_filter_size*2, initial_filter_size, 2, stride=2)

        self.expand_1_1 = self.expand(initial_filter_size*2, initial_filter_size)
        self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size)
        # segmentation head
        self.final = nn.Conv2d(initial_filter_size, num_classes, kernel_size=1)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    @staticmethod
    def center_crop(layer, target_width, target_height):
        _, _, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x):
        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        contr_3 = self.drop_enc3(contr_3)          # dropout on the deep skip + downstream
        pool = self.pool(contr_3)

        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        contr_4 = self.drop_enc4(contr_4)          # dropout on the deep skip + downstream
        pool = self.pool(contr_4)

        center = self.center(pool)

        crop = self.center_crop(contr_4, center.size()[2], center.size()[3])
        concat = torch.cat([center, crop], 1)
        expand = self.expand_4_2(self.expand_4_1(concat))
        expand = self.drop_dec4(expand)            # dropout in the deep decoder block
        upscale = self.upscale4(expand)

        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop], 1)
        expand = self.expand_3_2(self.expand_3_1(concat))
        expand = self.drop_dec3(expand)            # dropout in the deep decoder block
        upscale = self.upscale3(expand)

        crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop], 1)
        expand = self.expand_2_2(self.expand_2_1(concat))
        upscale = self.upscale2(expand)

        crop = self.center_crop(contr_1, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop], 1)
        expand = self.expand_1_2(self.expand_1_1(concat))

        return self.final(expand)
