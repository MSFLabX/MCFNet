import torch
import torch.nn as nn
import torch.nn.functional as F
from DWT import DWT_2D
from FDFA import FDFA, Cross_atten
import math

def img2seq(x):
    [b, c, h, w] = x.shape
    x = x.reshape((b, c, h*w))
    return x


def seq2img(x):
    [b, c, d] = x.shape
    p = int(d ** .5)
    x = x.reshape((b, c, p, p))
    return x

class MCFNet(nn.Module):
    def __init__(self, l1, l2, patch_size, num_classes, fae_embed_dim, attn_kernel_size, wavename):
        super().__init__()
        self.wave_hsi = HSI_Wave(wavename=wavename, in_channels=l1, out_channels=fae_embed_dim,
                                             attn_kernel_size=attn_kernel_size, patch_size=patch_size)
        self.wave_lidar = LiDAR_Wave(wavename=wavename, in_channels=l2, out_channels=fae_embed_dim,
                                           attn_kernel_size=attn_kernel_size, patch_size=patch_size)
        
        self.query_common = nn.Sequential(
            nn.Conv2d(in_channels=fae_embed_dim, out_channels=1, kernel_size=3,stride=1, padding=1),
            nn.Softmax(dim=1)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(l1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(l2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(32, fae_embed_dim, 3, 1, 1),
            nn.BatchNorm2d(fae_embed_dim),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, fae_embed_dim, 3, 1, 1),
            nn.BatchNorm2d(fae_embed_dim),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(32, fae_embed_dim, 3, 1, 1),
            nn.BatchNorm2d(fae_embed_dim),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, fae_embed_dim, 3, 1, 1),
            nn.BatchNorm2d(fae_embed_dim),
            nn.ReLU(),  # No effect on order
            nn.MaxPool2d(2),
        )

        self.linear1 = nn.Linear(((patch_size // 2) * 1) ** 2, patch_size ** 2)
        self.linear2 = nn.Linear(((patch_size // 2) * 2) ** 2, patch_size ** 2)

        self.cross_attn = Cross_atten(fae_embed_dim, num_heads=4)

        self.classifier = Classifier(Classes=num_classes, cls_embed_dim=fae_embed_dim)

    def fusion(self, hsi, lidar):
        weight_common = self.query_common(hsi)

        common_embeds = weight_common * hsi
        sep_hsi_embeds = hsi - 1.2 * common_embeds
        sep_lidar_embeds = lidar - 1.2 * common_embeds
        x_cnn = sep_hsi_embeds + sep_lidar_embeds

        return x_cnn



    def forward(self, img11, img21, img12, img22, img13, img23, img14, img24):

        x11 = self.conv1(img11)
        x21 = self.conv2(img21)
        x12 = self.conv1(img12)
        x22 = self.conv2(img22)
        x1_1 = self.conv1_1(x11)
        x2_1 = self.conv2_1(x21)
        x1_2 = self.conv1_2(x12)
        x2_2 = self.conv2_2(x22)

        hsi11 = self.wave_hsi(img11)
        lidar21 = self.wave_lidar(img21)

        hsi12 = self.wave_hsi(img12)
        lidar22 = self.wave_lidar(img22)

        # multimodal fusion
        x_cnn_1 = self.fusion(hsi11, lidar21)
        x_cnn_2 = self.fusion(x1_1, x2_1)
        x_cnn_3 = self.fusion(hsi12, lidar22)
        x_cnn_4 = self.fusion(x1_2, x2_2)

        # multiscales fusion
        x_flat1 = self.cross_attn(x_cnn_1, x_cnn_2)
        x_flat2 = self.cross_attn(x_cnn_3, x_cnn_4)

        # cross-domain fusion
        x_flat1 = x_flat1.flatten(2)
        x_flat2 = x_flat2.flatten(2)

        x_1 = self.linear1(x_flat1)
        x_2 = self.linear2(x_flat2)
        x_cnn = x_1 + x_2

        p = 8

        x_cnn = x_cnn.reshape((x_cnn.shape[0], x_cnn.shape[1], p, p))
        x_cls = self.classifier(x_cnn)
        
        return x_cls

# wavelet transform for HSI
class HSI_Wave(nn.Module):
    def __init__(self, wavename, in_channels, out_channels=64, attn_kernel_size=7, patch_size=4):
        super(HSI_Wave, self).__init__()
        self.DWT_layer_2D = DWT_2D(wavename=wavename)

        # 2d cnn for x_ll
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.attn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=attn_kernel_size, stride=1,
                      padding=attn_kernel_size // 2),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        # high frequency components processing
        self.conv_high = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # 2d cnn for all components
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.fdfa = FDFA(out_channels, num_heads=4)

    def forward(self, hsi_img):
        x_dwt = self.DWT_layer_2D(hsi_img)

        # x_ll -> ch_select
        x_ll = x_dwt[0]
        x_ll = self.conv1(x_ll)
        x_ll = self.conv2(x_ll)
        x_ll_1 = torch.cat([x_ll.mean(1, keepdim=True), x_ll.max(1, keepdim=True)[0]], dim=1)
        x_ll_2 = self.attn(x_ll_1)
        x_ll = x_ll * x_ll_2.expand_as(x_ll)
        # high frequency component processing
        x_high = torch.cat([x for x in x_dwt[1:4]], dim=1)
        x_high = self.conv_high(x_high)
        # FDFA
        x = self.fdfa(x_high,x_ll)


        return x

# wavelet transform for LiDAR
class LiDAR_Wave(nn.Module):
    def __init__(self, wavename, in_channels=1, out_channels=64, attn_kernel_size=7,patch_size=4):
        super(LiDAR_Wave, self).__init__()
        self.DWT_layer_2D = DWT_2D(wavename=wavename)

        # 2d cnn for x_ll
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=attn_kernel_size, stride=1, padding=attn_kernel_size // 2),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        # 2d cnn for high components
        self.conv_high = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # 2d cnn for all components
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.fdfa = FDFA(out_channels, num_heads=4)

    def forward(self, lidar_img):
        x_dwt = self.DWT_layer_2D(lidar_img)

        # x_ll -> Conv2d layer & spatial attn
        x_ll = x_dwt[0]
        x_ll = self.conv1(x_ll)
        x_ll = self.conv2(x_ll)
        x_ll_1 = torch.cat([x_ll.mean(1, keepdim=True), x_ll.max(1, keepdim=True)[0]], dim=1)
        x_ll_2 = self.attn(x_ll_1)
        x_ll = x_ll * x_ll_2.expand_as(x_ll)

        # high frequency component processing
        x_high = torch.cat([x for x in x_dwt[1:4]], dim=1)
        x_high = self.conv_high(x_high)
        # FDFA
        x = self.fdfa(x_high,x_ll)
        return x

class Classifier(nn.Module):
    def __init__(self, Classes, cls_embed_dim):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=cls_embed_dim, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.linear = nn.Linear(in_features=32, out_features=Classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x_out = F.softmax(x, dim=1)

        return x_out
