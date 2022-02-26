import torch.nn as nn


class generator(nn.Module):
    def __init__(self, input_size, datasets):
        super().__init__()

        self.input_size = input_size
        self.datasets = datasets

        self.input_layer = self.linear_input(self.input_size)

        if self.datasets == 'can':
            self.conv_up1 = self.conv_upsample_block(2048, 2048, 1, 1, 0)
            self.conv_up2 = self.conv_upsample_block(2048, 1024, 2, 2, 0)
            self.conv_up3 = self.conv_upsample_block(1024, 512, 2, 2, 0)
            self.conv_up4 = self.conv_upsample_block(512, 256, 2, 2, 0)
            self.conv_up5 = self.conv_upsample_block(256, 128, 2, 2, 0)
            self.conv_up6 = self.conv_upsample_block(128, 64, 2, 2, 0)
            self.conv_upout = nn.Sequential(
                nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=False),
                nn.Tanh()
            )
        
        elif self.datasets == 'cifar10':
            self.conv_up1 = self.conv_upsample_block(2048, 2048, 1, 1, 0)
            self.conv_up2 = self.conv_upsample_block(2048, 1024, 1, 1, 0)
            self.conv_up3 = self.conv_upsample_block(1024, 512, 1, 1, 0)
            self.conv_up4 = self.conv_upsample_block(512, 256, 1, 1, 0)
            self.conv_up5 = self.conv_upsample_block(256, 128, 2, 2, 0)
            self.conv_up6 = self.conv_upsample_block(128, 64, 2, 2, 0)
            self.conv_upout = nn.Sequential(
                nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=False),
                nn.Tanh()
            )

    def forward(self, input_noise):
        out = self.input_layer(input_noise)
        out = out.view(-1, 2048, 4, 4)
        out = self.conv_up1(out)
        out = self.conv_up2(out)
        out = self.conv_up3(out)
        out = self.conv_up4(out)
        out = self.conv_up5(out)
        out = self.conv_up6(out)
        out = self.conv_upout(out)

        return out

    def linear_input(self, input_size):
        net = nn.Sequential(
            nn.Linear(input_size, 2048*4*4, bias=False),
            nn.BatchNorm1d(2048*4*4),
            nn.ReLU(inplace=True),
        )
        return net

    def conv_upsample_block(self, in_channels, out_channels, kernel_size, stride, padding):
        net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return net