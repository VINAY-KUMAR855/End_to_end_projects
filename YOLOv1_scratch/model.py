'''
Implementation yolov1 architecture
with slight modification with added BatchNorm
'''

import torch
import torch.nn as nn

# architecture_config
architecture_config = [
    # (kernal_size, number of filerts as output, stride, padding)
    (7, 64, 2, 3),
    "M", # "M" is simply maxpooling with stride 2x2 and kernel 2x2
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [
        # (kernal_size, number of filerts as output, stride, padding), int with number of repeats
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        4
    ],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [
        (1, 512, 1, 0), 
        (3, 1024, 1, 1), 
        2
    ],
    (3, 1024, 1,1),
    (3, 1024, 2, 1),
    (3, 1024, 1,1),
    (3, 1024, 1,1)
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class YOLOv1(nn.Module):
    def __init__(self,in_channels=3, split_size=7, num_boxes=2, num_classes=20, **kwargs):
        super().__init__()
        self.architectue = architecture_config
        self.in_channels = in_channels
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.darknet = self._create_conv_layers(self.architectue)
        self.fcs = self._create_fcs()

    def forward(self, x):
        x = self.darknet(x)
        x = self.fcs(x) # [batch, 1470]
        x = x.reshape(
            -1, # Batch
            self.S,
            self.S,
            self.B * 5 + self.C
        ) # [batch, 7, 7, 30]
        return x

    def _create_conv_layers(self, architectue):
        layers = []
        in_channels = self.in_channels
        for x in architectue:
            if isinstance(x, tuple):
                layers.append(
                    CNNBlock(
                        in_channels, x[1], kernel_size = x[0], stride=x[2], padding=x[3]
                    )
                )
                in_channels = x[1]
            elif x=="M":
                layers.append(
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            elif isinstance(x, list):
                conv1 = x[0] # Tuple
                conv2 = x[1] # Tuple
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers.append(
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size = conv1[0],
                            stride = conv1[2],
                            padding = conv1[3]
                        )
                    )
                    layers.append(
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size = conv2[0],
                            stride = conv2[2],
                            padding = conv2[3]
                        )
                    )
                    in_channels = conv2[1]
        return nn.Sequential(*layers)
    def _create_fcs(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*self.S * self.S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S*self.S*(self.C+self.B*5)) # (S,S,30) where C+B*5 = 30
        )

if __name__ == "__main__":

    model = YOLOv1()

    x = torch.randn(2, 3, 448, 448)

    output = model(x)

    print("Input :", x.shape)
    print("Output:", output.shape)