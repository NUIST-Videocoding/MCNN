import time

import torch
import torch.nn.functional as F
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.models import MeanScaleHyperprior
from torch import nn

from src.utils.stream_helper import *

class Atrous_Block(nn.Module):
    def __init__(self, M, N, dilation):
        super(Atrous_Block, self).__init__()
        self.dilation = dilation

        self.kernel_size = 3
        self.conv1 = nn.Conv2d(M, N, 1)
        self.atrous1 = nn.Sequential(
            nn.Conv2d(N, N, self.kernel_size, stride=1, padding=1, dilation=self.dilation[0]),
            nn.LeakyReLU(inplace=True),
        )
        self.atrous2 = nn.Sequential(
            nn.Conv2d(N, N, self.kernel_size, stride=1, padding=self.dilation[1], dilation=self.dilation[1]),
            nn.LeakyReLU(inplace=True),
        )
        self.atrous3 = nn.Sequential(
            nn.Conv2d(N, N, self.kernel_size, stride=1, padding=self.dilation[2], dilation=self.dilation[2]),
            nn.LeakyReLU(inplace=True),
        )
        self.atrous4 = nn.Sequential(
            nn.Conv2d(N, N, self.kernel_size, stride=1, padding=self.dilation[3], dilation=self.dilation[3]),
            nn.LeakyReLU(inplace=True),
        )

        self.conv2 = nn.Conv2d(N * 4, N, 1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):  # (M, y_layer   N, p_layer)
        x = self.conv1(x)
        identify = x
        x1 = self.atrous1(x)
        x2 = self.atrous2(x)
        x3 = self.atrous3(x)
        x4 = self.atrous4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = x + identify

        return x


class interaction(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super(interaction, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1)
        self.res_in_ch = Atrous_Block(out_ch, out_ch, dilation)
        self.res_out_ch = Atrous_Block(out_ch, out_ch, dilation)
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1)
        )

        self.conv4 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1, stride=1)
        self.soft_1 = nn.Sigmoid()
        self.soft_2 = nn.Sigmoid()

    def forward(self, f_p, f_y):  # (M, y_layer   N, p_layer)
        f_y = self.conv2(f_y)
        f_p = self.conv1(f_p)
        a = f_p + f_y
        f_p_1 = self.res_in_ch(f_p)
        f_y_1 = self.res_out_ch(f_y)
        f_a = torch.cat((f_p_1, f_y_1), dim=1)
        att_1 = self.soft_1(self.conv3_1(f_a))
        att_2 = self.soft_2(self.conv3_2(f_a))
        f_ay = att_1 * f_y_1
        f_ap = att_2 * f_p_1
        f = self.conv4(f_ay + f_ap)
        f = f + a
        return f

class FeatureCombine(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        # in_ch,out_ch
        self.dilation1 = [1, 2, 4, 8]  # p5     16*16
        self.dilation2 = [1, 2, 4, 8]  # p4     32*32
        self.dilation3 = [1, 2, 4, 8]  # p3     64*64
        self.dilation4 = [1, 2, 4, 8]  # p2     128*128

        self.p2Encoder = nn.Sequential(
            ResidualBlockWithStride(in_ch, out_ch, stride=2),
            # ResidualBlock(in_ch, out_ch),
        )
        self.interact1 = interaction(in_ch, out_ch, self.dilation3)

        self.p3Encoder = nn.Sequential(
            ResidualBlockWithStride(out_ch, out_ch, stride=2),
            # ResidualBlock(in_ch, out_ch),
        )
        self.interact2 = interaction(in_ch, out_ch, self.dilation2)

        self.p4Encoder = nn.Sequential(
            ResidualBlockWithStride(out_ch, out_ch, stride=2),
            # ResidualBlock(in_ch, out_ch),
        )
        self.interact3 = interaction(in_ch, out_ch, self.dilation1)

        self.p5Encoder = nn.Sequential(
            ResidualBlockWithStride(out_ch, out_ch, stride=2),
            ResidualBlock(out_ch, out_ch),
        )

    def forward(self, p_layer_features):
        # p_layer_features contains padded features p2, p3, p4, p5
        p2, p3, p4, p5 = tuple(p_layer_features)
        y = self.p2Encoder(p2)
        y = self.p3Encoder(self.interact1(p3, y))
        y = self.p4Encoder(self.interact2(p4, y))
        y = self.p5Encoder(self.interact3(p5, y))

        return y



class FeatureSynthesis(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.dilation1 = [1, 2, 4, 8]  # p5     16*16
        self.dilation2 = [1, 2, 4, 8]  # p4     32*32
        self.dilation3 = [1, 2, 4, 8]  # p3     64*64
        self.dilation4 = [1, 2, 4, 8]  # p2     128*128


        self.y5Upsample = ResidualBlockUpsample(in_ch, in_ch, 2)

        self.y4Upsample = ResidualBlockUpsample(in_ch, in_ch, 4)

        self.y3Upsample = ResidualBlockUpsample(in_ch, in_ch, 8)

        # self.p2Upsample = ResidualBlockUpsample(out_ch, out_ch, 2)

        self.p5Decoder = nn.Sequential(
            ResidualBlock(in_ch, out_ch),
            ResidualBlockUpsample(out_ch, out_ch, 2),
        )
        self.p4Fusion = interaction(in_ch, out_ch, self.dilation1)
        self.p4Decoder = nn.Sequential(
            # ResidualBlock(out_ch, out_ch),
            ResidualBlockUpsample(out_ch, out_ch, 2),
        )
        self.p3Fusion = interaction(in_ch, out_ch, self.dilation2)
        self.p3Decoder = nn.Sequential(
            # ResidualBlock(out_ch, out_ch),
            ResidualBlockUpsample(out_ch, out_ch, 2),
        )
        self.p2Fusion = interaction(in_ch, out_ch, self.dilation3)
        self.p2Decoder = nn.Sequential(
            # ResidualBlock(out_ch, out_ch),
            ResidualBlockUpsample(out_ch, out_ch, 2),
        )

    def forward(self, y_hat):
        p5 = self.p5Decoder(y_hat)

        y_hat_up1 = self.y5Upsample(y_hat)
        y_hat_up2 = self.y4Upsample(y_hat)
        y_hat_up3 = self.y3Upsample(y_hat)

        p4 = self.p4Fusion(y_hat_up1, p5)
        p4 = self.p4Decoder(p4)

        p3 = self.p3Fusion(y_hat_up2, p4)
        p3 = self.p3Decoder(p3)

        p2 = self.p2Fusion(y_hat_up3, p3)
        p2 = self.p2Decoder(p2)

        return [p2, p3, p4, p5]


class FeatureCompressor(MeanScaleHyperprior):
    def __init__(self, y_ch=192, f_ch=256, **kwargs):
        super().__init__(y_ch, f_ch, **kwargs)

        self.g_a = FeatureCombine(f_ch, y_ch)
        self.g_s = FeatureSynthesis(y_ch, f_ch)

        self.h_a = nn.Sequential(
            conv3x3(y_ch, y_ch),
            nn.LeakyReLU(inplace=True),
            conv3x3(y_ch, y_ch),
            nn.LeakyReLU(inplace=True),
            conv3x3(y_ch, y_ch, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(y_ch, y_ch),
            nn.LeakyReLU(inplace=True),
            conv3x3(y_ch, y_ch, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(y_ch, y_ch),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(y_ch, y_ch, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(y_ch, y_ch * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(y_ch * 3 // 2, y_ch * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(y_ch * 3 // 2, y_ch * 2),
        )

        self.p6Decoder = nn.Sequential(nn.MaxPool2d(1, stride=2))

    def forward(self, features):  # features: [p2, p3, p4, p5, p6]
        features = features[:-1]
        _, _, p2_h, p2_w = features[0].shape
        pad_info = self.cal_feature_padding_size((p2_h, p2_w))
        features = self.feature_padding(features, pad_info)

        y = self.g_a(features)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        recon_p_layer_features = self.g_s(y_hat)
        recon_p_layer_features = self.feature_unpadding(
            recon_p_layer_features, pad_info
        )

        p6 = self.p6Decoder(
            recon_p_layer_features[3]
        )  # p6 is generated from p5 directly

        recon_p_layer_features.append(p6)

        return {
            "features": recon_p_layer_features,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, features):  # features: [p2, p3, p4, p5, p6]
        features = features[:-1]
        _, _, p2_h, p2_w = features[0].shape
        pad_info = self.cal_feature_padding_size((p2_h, p2_w))
        features = self.feature_padding(features, pad_info)
        y = self.g_a(features)

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, p2_h, p2_w):
        assert isinstance(strings, list) and len(strings) == 2
        pad_info = self.cal_feature_padding_size((p2_h, p2_w))
        padded_p2_h = pad_info["padded_size"][0][0]
        padded_p2_w = pad_info["padded_size"][0][1]
        z_shape = get_downsampled_shape(padded_p2_h, padded_p2_w, 64)
        z_hat = self.entropy_bottleneck.decompress(strings[1], z_shape)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )

        recon_p_layer_features = self.g_s(y_hat)
        recon_p_layer_features = self.feature_unpadding(recon_p_layer_features)
        p6 = self.p6Decoder(
            recon_p_layer_features[3]
        )  # p6 is generated from p5 directly
        recon_p_layer_features.append(p6)
        return {"features": recon_p_layer_features}

    def encode_decode(self, features, output_path, p2_height, p2_width):
        encoding_time_start = time.time()
        encoded = self.encode(features, output_path, p2_height, p2_width)
        encoding_time = time.time() - encoding_time_start
        decoding_time_start = time.time()
        decoded = self.decode(output_path)
        decoding_time = time.time() - decoding_time_start
        encoded.update(decoded)
        encoded["encoding_time"] = encoding_time
        encoded["decoding_time"] = decoding_time
        return encoded

    def encode(self, features, output_path, p2_height, p2_width):
        encoded = self.compress(features)
        y_string = encoded["strings"][0][0]
        z_string = encoded["strings"][1][0]

        encode_feature(p2_height, p2_width, y_string, z_string, output_path)
        bits = filesize(output_path) * 8
        summary = {
            "bit": bits,
            "bit_y": len(y_string) * 8,
            "bit_z": len(z_string) * 8,
        }
        encoded.update(summary)
        return encoded

    def decode(self, input_path):
        p2_height, p2_width, y_string, z_string = decode_feature(input_path)
        decoded = self.decompress([y_string, z_string], p2_height, p2_width)
        return decoded

    def cal_feature_padding_size(self, p2_shape):
        ps_list = [64, 32, 16, 8]
        ori_size = []
        paddings = []
        unpaddings = []
        padded_size = []

        ori_size.append(p2_shape)
        for i in range(len(ps_list) - 1):
            h, w = ori_size[-1]
            ori_size.append(((h + 1) // 2, (w + 1) // 2))

        for i, ps in enumerate(ps_list):
            h = ori_size[i][0]
            w = ori_size[i][1]

            h_pad_len = ps - h % ps if h % ps != 0 else 0
            w_pad_len = ps - w % ps if w % ps != 0 else 0

            paddings.append(
                (
                    w_pad_len // 2,
                    w_pad_len - w_pad_len // 2,
                    h_pad_len // 2,
                    h_pad_len - h_pad_len // 2,
                )
            )
            unpaddings.append(
                (
                    0 - (w_pad_len // 2),
                    0 - (w_pad_len - w_pad_len // 2),
                    0 - (h_pad_len // 2),
                    0 - (h_pad_len - h_pad_len // 2),
                )
            )

        for i, p in enumerate(paddings):
            h = ori_size[i][0]
            w = ori_size[i][1]
            h_pad_len = p[2] + p[3]
            w_pad_len = p[0] + p[1]
            padded_size.append((h + h_pad_len, w + w_pad_len))

        return {
            "ori_size": ori_size,
            "paddings": paddings,
            "unpaddings": unpaddings,
            "padded_size": padded_size,
        }

    def feature_padding(self, features, pad_info):
        p2, p3, p4, p5 = features
        paddings = pad_info["paddings"]

        p2 = F.pad(p2, paddings[0], mode="reflect")
        p3 = F.pad(p3, paddings[1], mode="reflect")
        p4 = F.pad(p4, paddings[2], mode="reflect")
        p5 = F.pad(p5, paddings[3], mode="reflect")
        return [p2, p3, p4, p5]

    def feature_unpadding(self, features, pad_info):
        p2, p3, p4, p5 = features
        unpaddings = pad_info["unpaddings"]

        p2 = F.pad(p2, unpaddings[0])
        p3 = F.pad(p3, unpaddings[1])
        p4 = F.pad(p4, unpaddings[2])
        p5 = F.pad(p5, unpaddings[3])
        return [p2, p3, p4, p5]
