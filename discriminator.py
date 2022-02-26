import torch
import torch.nn as nn


class discriminator(nn.Module):
    def __init__(self, c_num, datasets):
        super().__init__()
        
        self.datasets = datasets
        self.c_num = c_num

        if self.datasets == 'can':
            self.conv_down1 = self.conv_downsample_block(3, 32)
            self.conv_down2 = self.conv_downsample_block(32, 64)
            self.conv_down3 = self.conv_downsample_block(64, 128)
            self.conv_down4 = self.conv_downsample_block(128, 256)
            self.conv_down5 = self.conv_downsample_block(256, 512)
            self.flatten = nn.Flatten()

            self.r_out = nn.Linear(512*8*8, 1, bias=False)

            self.c_linear1 = self.relu_linear(512*8*8, 1024)
            self.c_linear2 = self.relu_linear(1024, 512)
            self.c_out = nn.Sequential(
                nn.Linear(512, self.c_num, bias=False),
                nn.Softmax(1)
            )

        elif self.datasets == 'cifar10':
            self.conv_down1 = self.conv_downsample_block(3, 32)
            self.conv_down2 = self.conv_downsample_block(32, 64)
            self.conv_down3 = self.conv_downsample_block(64, 128)
            self.conv_down4 = self.conv_downsample_block(128, 256, 2)
            self.conv_down5 = self.conv_downsample_block(256, 512, 2)
            self.flatten = nn.Flatten()

            self.r_out = nn.Linear(512*2*2, 1, bias=False)

            self.c_linear1 = self.relu_linear(512*2*2, 1024)
            self.c_linear2 = self.relu_linear(1024, 512)
            self.c_out = nn.Sequential(
                nn.Linear(512, self.c_num, bias=False),
                nn.Softmax()
            )

    def forward(self, input_image):
        out = self.conv_down1(input_image)
        out = self.conv_down2(out)
        out = self.conv_down3(out)
        out = self.conv_down4(out)
        out = self.conv_down5(out)
        out = self.flatten(out)

        r_out = self.r_out(out)

        c_out = self.c_linear1(out)
        c_out = self.c_linear2(c_out)
        c_out = self.c_out(c_out)

        return torch.sigmoid(r_out), c_out

    def conv_downsample_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,):
        net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        return net

    def relu_linear(self, in_channels, out_channels):
        net = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        return net