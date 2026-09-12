import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsamplerBlock(nn.Module):
    """
    Downsamples the feature maps by concatenating a 2x2 MaxPool (stride 2)
    and a 3x3 Convolution (stride 2) to preserve information efficiently.
    """
    def __init__(self, ninput, noutput):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2,stride=2)
        self.conv = nn.Conv2d(ninput,noutput-ninput,kernel_size=3,stride=2,padding=1,bias=True)
        self.bn = nn.BatchNorm2d(noutput,eps=1e-3)

    def forward(self, x):
        output = torch.cat([self.maxpool(x),self.conv(x)],1)
        output = self.bn(output)
        return F.relu(output)
    
class non_bottleneck_1d(nn.Module):
    """
    It Factorizes a 3x3 2D convolution into 
    sequential 3x1 and 1x3 1D convolutions to reduce parameters and latency.
    Supports dilation for wider receptive fields and dropout for regularization.
    """
    def __init__(self, ch, dropout, dilation):
        super().__init__()
        # First factorized 1D convolution pair
        self.conv3x1_1 = nn.Conv2d(ch,ch,kernel_size=(3,1), stride=1, padding=(1,0),bias=True)
        self.conv1x3_1 = nn.Conv2d(ch,ch,kernel_size=(1,3), stride=1, padding=(0,1),bias=True)
        self.bn1 = nn.BatchNorm2d(ch, eps=1e-3)
        # second factorized 1D convolution pair
        self.conv3x1_2 = nn.Conv2d(ch,ch,kernel_size=(3,1), stride=1, padding=(1*dilation,0),bias=True,dilation=(dilation,1))
        self.conv1x3_2 = nn.Conv2d(ch,ch,kernel_size=(1,3), stride=1, padding=(0, 1*dilation),bias=True,dilation=(1, dilation))
        self.bn2 = nn.BatchNorm2d(ch, eps=1e-3)

        self.dropout = nn.Dropout(dropout) if dropout>0 else None

    def forward(self, x):
        residual = x
        # first layer block
        output = self.conv3x1_1(x)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)
        # second layer block
        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        if self.dropout is not None:
            output = self.dropout(output)
        # residual skip connection
        return F.relu(output+residual)

class Encoder(nn.Module):
    """
    ERFNet Encoder: downsamples resolution while building strong semantic features.
    """
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList()

        # Layer 1&2 : downsampling
        self.layers.append(DownsamplerBlock(3,16))
        self.layers.append(DownsamplerBlock(16,64))
        # Layers 3-7: non_bottleneck_1d blocks without dilation
        for _ in range(5):
            self.layers.append(non_bottleneck_1d(64,dropout=0.03, dilation=1))
        # Layer 8 : final down sampling
        self.layers.append(DownsamplerBlock(64, 128))
        # Layers 9-16: Dilated non_bottleneck_1d blocks (alternating dilation factors)
        for _ in range(2): # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, kernel_size=3, stride=2,padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        return F.relu(output)

class Decoder(nn.Module):
    """
    ERFNet Decoder: Upsamples the encoder feature maps back to original resolution.
    """
    def __init__(self, channels):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64,0,1))
        self.layers.append(non_bottleneck_1d(64,0,1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16,0,1))
        self.layers.append(non_bottleneck_1d(16,0,1))

        # self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.output_conv = nn.Conv2d(16,channels,kernel_size=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_conv(x)

class ERFNet(nn.Module):
    """
    Full ERFNet architecture combining Encoder and Decoder modules.
    """
    def __init__(self, channels):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(channels)

    def forward(self, x):
        output = self.encoder(x)
        return self.decoder(output)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ERFNet(channels=96).to(device)
    x = torch.rand(size=(1, 3, 256, 512), device=device)
    res = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", res.shape)