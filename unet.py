# -*- coding: utf-8 -*-

# Imports
import torch
import torch.nn as nn

def sinusoidal_embedding(n, d):
    """
    Generates a sinusoidal embedding of size n x d.

    Parameters:
        n (int): The number of rows in the embedding.
        d (int): The number of columns in the embedding.

    Returns:
        torch.Tensor: The sinusoidal embedding of size n x d.
    """
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding

class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        """
        Initializes a new instance of the MyBlock class.

        Parameters:
            shape (tuple): The shape of the input tensor.
            in_c (int): The number of input channels.
            out_c (int): The number of output channels.
            kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
            stride (int, optional): The stride of the convolutional kernel. Defaults to 1.
            padding (int, optional): The padding of the convolutional kernel. Defaults to 1.
            activation (nn.Module, optional): The activation function to use. Defaults to None.
            normalize (bool, optional): Whether to apply layer normalization. Defaults to True.
        """
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.GELU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.activation(out)  
              
        return out

    def _make_te(self, dim_in, dim_out):
        """
        Creates a sequential neural network model with two linear layers and a SiLU activation function.

        Args:
            dim_in (int): The number of input features.
            dim_out (int): The number of output features.

        Returns:
            torch.nn.Sequential: The sequential neural network model.
        """
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.Linear(dim_out, dim_out)
        )

class MyUNet(nn.Module):
    def __init__(self, channel, size, n_steps=500, time_emb_dim=100):
        """
        Initializes the MyUNet model, a U-Net architecture with customizable configurations for image segmentation tasks.
        
        This constructor initializes various layers and components based on the specified image size and configurations.
        It sets up a U-Net architecture with optional depth based on the 'size' parameter which can be 128, 256 or 512pixels,
        and constructs blocks and transitional layers correspondingly. The architecture uses sinusoidal embeddings for time
        encoding and multiple encoding and decoding blocks with convolutional and transposed convolutional layers.
        
        Parameters:
        - channel (int): The number of channels in the input and output images. Typically, this corresponds to the
        number of color channels (e.g., 3 for RGB images).
        - size (int): The dimension of the input images (e.g., 128 or 256 pixels). This size determines the depth and
        complexity of the U-Net architecture.
        - n_steps (int, optional): The number of steps or positions for the time embedding, with a default of 500. This
        is used in the sinusoidal time embedding.
        - time_emb_dim (int, optional): The dimensionality of the time embedding vector, with a default of 100.
        
        Raises:
        - ValueError: If the 'size' parameter is not recognized (i.e., not 128 or 256), a ValueError is raised.
        
        The network comprises several stages of downsampling followed by upsampling to create a symmetric architecture,
        with additional bottleneck layers. Time-dependent features are integrated at multiple levels of the network.
        """
        super(MyUNet, self).__init__()
        self.size = size

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        
        
        if self.size == 128:
            # First half
            self.te1 = self._make_te(time_emb_dim, channel)
            self.b1 = nn.Sequential(
                MyBlock((channel, 128, 128), channel, 16),
                MyBlock((16, 128, 128), 16, 32),
                MyBlock((32, 128, 128), 32, 32)
            )
            self.down1 = nn.Conv2d(32, 32, 4, 2, 1) # 128x128 --> 64x64

            self.te2 = self._make_te(time_emb_dim, 32)
            self.b2 = nn.Sequential(
                MyBlock((32, 64, 64), 32, 64),
                MyBlock((64, 64, 64), 64, 64),
                MyBlock((64, 64, 64), 64, 64)
            )
            self.down2 = nn.Conv2d(64, 64, 4, 2, 1) # 64x64 --> 32x32

            self.te3 = self._make_te(time_emb_dim, 64)
            self.b3 = nn.Sequential(
                MyBlock((64, 32, 32), 64, 128),
                MyBlock((128, 32, 32), 128, 128),
                MyBlock((128, 32, 32), 128, 128)
            )
            self.down3 = nn.Conv2d(128, 128, 4, 2, 1) # 32x32 --> 16x16

            self.te4 = self._make_te(time_emb_dim, 128)
            self.b4 = nn.Sequential(
                MyBlock((128, 16, 16), 128, 256),
                MyBlock((256, 16, 16), 256, 256),
                MyBlock((256, 16, 16), 256, 256)
            )
            self.down4 = nn.Conv2d(256, 256, 4, 2, 1) # 16x16 --> 8x8

            self.te5 = self._make_te(time_emb_dim, 256)
            self.b5 = nn.Sequential(
                MyBlock((256, 8, 8), 256, 512),
                MyBlock((512, 8, 8), 512, 512),
                MyBlock((512, 8, 8), 512, 512)
            )
            self.down5 = nn.Sequential(
                nn.Conv2d(512, 512, 2, 1), # 8x8 --> 7x7
                nn.GELU(),
                nn.Conv2d(512, 512, 4, 2, 2) # 7x7 --> 4x4
            )

            # Bottleneck
            self.te_mid = self._make_te(time_emb_dim, 512)
            self.b_mid = nn.Sequential(
                MyBlock((512, 4, 4), 512, 512),
                MyBlock((512, 4, 4), 512, 512),
                MyBlock((512, 4, 4), 512, 512)
            )

            # Second half
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(512, 512, 3, 2, 1), #4x4 --> 7x7
                nn.GELU(),
                nn.ConvTranspose2d(512, 512, 2, 1) #7x7 --> 8x8
            )

            self.te6 = self._make_te(time_emb_dim, 1024) # 512 (out5) + 512 = 1024
            self.b6 = nn.Sequential(
                MyBlock((1024, 8, 8), 1024, 512),
                MyBlock((512, 8, 8), 512, 256),
                MyBlock((256, 8, 8), 256, 256)
            )

            self.up2 = nn.ConvTranspose2d(256, 256, 4, 2, 1) # 8x8 --> 16x16
            self.te7 = self._make_te(time_emb_dim, 512) # 256 (out4) + 256 = 512
            self.b7 = nn.Sequential(
                MyBlock((512, 16, 16), 512, 256),
                MyBlock((256, 16, 16), 256, 128),
                MyBlock((128, 16, 16), 128, 128)
            )

            self.up3 = nn.ConvTranspose2d(128, 128, 4, 2, 1) # 16x16 --> 32x32
            self.te8 = self._make_te(time_emb_dim, 256) # 128 (out3) + 128 = 256
            self.b8 = nn.Sequential(
                MyBlock((256, 32, 32), 256, 128),
                MyBlock((128, 32, 32), 128, 64),
                MyBlock((64, 32, 32), 64, 64, normalize=False)
            )

            self.up4 = nn.ConvTranspose2d(64, 64, 4, 2, 1) # 32x32 --> 64x64
            self.te9 = self._make_te(time_emb_dim, 128) # 64 (out2) + 64 = 128
            self.b9 = nn.Sequential(
                MyBlock((128, 64, 64), 128, 64),
                MyBlock((64, 64, 64), 64, 32),
                MyBlock((32, 64, 64), 32, 32, normalize=False)
            )

            self.up5 = nn.ConvTranspose2d(32, 32, 4, 2, 1) # 64x64 --> 128x128
            self.te10 = self._make_te(time_emb_dim, 64) # 32 (out1) + 32 = 64  
            self.b10 = nn.Sequential(
                MyBlock((64, 128, 128), 64, 32),
                MyBlock((32, 128, 128), 32, 16),
                MyBlock((16, 128, 128), 16, 16, normalize=False)
            )

            self.conv_out = nn.Conv2d(16, channel, 3, 1, 1)
        
        elif self.size == 256:
            # First half
            self.te1 = self._make_te(time_emb_dim, channel)
            self.b1 = nn.Sequential(
                MyBlock((channel, 256, 256), channel, 16),
                MyBlock((16, 256, 256), 16, 32),
                MyBlock((32, 256, 256), 32, 32)
            )
            self.down1 = nn.Conv2d(32, 32, 4, 2, 1) # 256x256 --> 128x128

            self.te2 = self._make_te(time_emb_dim, 32)
            self.b2 = nn.Sequential(
                MyBlock((32, 128, 128), 32, 64),
                MyBlock((64, 128, 128), 64, 64),
                MyBlock((64, 128, 128), 64, 64)
            )
            self.down2 = nn.Conv2d(64, 64, 4, 2, 1) # 128x128 --> 64x64

            self.te3 = self._make_te(time_emb_dim, 64)
            self.b3 = nn.Sequential(
                MyBlock((64, 64, 64), 64, 128),
                MyBlock((128, 64, 64), 128, 128),
                MyBlock((128, 64, 64), 128, 128)
            )
            self.down3 = nn.Conv2d(128, 128, 4, 2, 1) # 64x64 --> 32x32

            self.te4 = self._make_te(time_emb_dim, 128)
            self.b4 = nn.Sequential(
                MyBlock((128, 32, 32), 128, 256),
                MyBlock((256, 32, 32), 256, 256),
                MyBlock((256, 32, 32), 256, 256)
            )
            self.down4 = nn.Conv2d(256, 256, 4, 2, 1) # 32x32 --> 16x16
            
            self.te5 = self._make_te(time_emb_dim, 256)
            self.b5 = nn.Sequential(
                MyBlock((256, 16, 16), 256, 512),
                MyBlock((512, 16, 16), 512, 512),
                MyBlock((512, 16, 16), 512, 512)
            )
            self.down5 = nn.Conv2d(512, 512, 4, 2, 1) # 16x16 --> 8x8

            self.te6 = self._make_te(time_emb_dim, 512)
            self.b6 = nn.Sequential(
                MyBlock((512, 8, 8), 512, 1024),
                MyBlock((1024, 8, 8), 1024, 1024),
                MyBlock((1024, 8, 8), 1024, 1024)
            )
            self.down6 = nn.Sequential(
                nn.Conv2d(1024, 1024, 2, 1), # 8x8 --> 7x7
                nn.GELU(),
                nn.Conv2d(1024, 1024, 4, 2, 2) # 7x7 --> 4x4
            )

            # Bottleneck
            self.te_mid = self._make_te(time_emb_dim, 1024)
            self.b_mid = nn.Sequential(
                MyBlock((1024, 4, 4), 1024, 1024),
                MyBlock((1024, 4, 4), 1024, 1024),
                MyBlock((1024, 4, 4), 1024, 1024)
            )

            # Second half
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(1024, 1024, 3, 2, 1), #4x4 --> 7x7
                nn.GELU(),
                nn.ConvTranspose2d(1024, 1024, 2, 1) #7x7 --> 8x8
            )

            self.te7 = self._make_te(time_emb_dim, 2048) # 1024 (out6) + 1024 = 2048
            self.b7 = nn.Sequential(
                MyBlock((2048, 8, 8), 2048, 1024),
                MyBlock((1024, 8, 8), 1024, 512),
                MyBlock((512, 8, 8), 512, 512)
            )

            self.up2 = nn.ConvTranspose2d(512, 512, 4, 2, 1) #8x8 --> 16x16
            self.te8 = self._make_te(time_emb_dim, 1024) # 512 (out5) + 512 = 1024
            self.b8 = nn.Sequential(
                MyBlock((1024, 16, 16), 1024, 512),
                MyBlock((512, 16, 16), 512, 256),
                MyBlock((256, 16, 16), 256, 256)
            )

            self.up3 = nn.ConvTranspose2d(256, 256, 4, 2, 1) #16x16 --> 32x32
            self.te9 = self._make_te(time_emb_dim, 512) # 256 (out4) + 256 = 512
            self.b9 = nn.Sequential(
                MyBlock((512, 32, 32), 512, 256),
                MyBlock((256, 32, 32), 256, 128),
                MyBlock((128, 32, 32), 128, 128, normalize=False)
            )

            self.up4 = nn.ConvTranspose2d(128, 128, 4, 2, 1) #32x32 --> 64x64
            self.te10 = self._make_te(time_emb_dim, 256) # 128 (out3) + 128 = 256
            self.b10 = nn.Sequential(
                MyBlock((256, 64, 64), 256, 128),
                MyBlock((128, 64, 64), 128, 64),
                MyBlock((64, 64, 64), 64, 64, normalize=False)
            )

            self.up5 = nn.ConvTranspose2d(64, 64, 4, 2, 1) #64x64 --> 128x128
            self.te11 = self._make_te(time_emb_dim, 128) # 64 (out2) + 64 = 128
            self.b11 = nn.Sequential(
                MyBlock((128, 128, 128), 128, 64),
                MyBlock((64, 128, 128), 64, 32),
                MyBlock((32, 128, 128), 32, 32, normalize=False)
            )
            
            self.up6 = nn.ConvTranspose2d(32, 32, 4, 2, 1) #64x64 --> 128x128
            self.te12 = self._make_te(time_emb_dim, 64) # 32 (out1) + 32 = 64
            self.b12 = nn.Sequential(
                MyBlock((64, 256, 256), 64, 32),
                MyBlock((32, 256, 256), 32, 16),
                MyBlock((16, 256, 256), 16, 16, normalize=False)
            )

            self.conv_out = nn.Conv2d(16, channel, 3, 1, 1)
            
        elif self.size == 512:
            # First half
            self.te1 = self._make_te(time_emb_dim, channel)
            self.b1 = nn.Sequential(
                MyBlock((channel, 512, 512), channel, 16), # 512x512 --> 256x256
                MyBlock((16, 512, 512), 16, 32),
                MyBlock((32, 512, 512), 32, 32)
            )
            self.down1 = nn.Conv2d(32, 32, 4, 2, 1) # 256x256 --> 128x128

            self.te2 = self._make_te(time_emb_dim, 32)
            self.b2 = nn.Sequential(
                MyBlock((32, 256, 256), 32, 64),
                MyBlock((64, 256, 256), 64, 64),
                MyBlock((64, 256, 256), 64, 64)
            )
            self.down2 = nn.Conv2d(64, 64, 4, 2, 1) # 128x128 --> 64x64

            self.te3 = self._make_te(time_emb_dim, 64)
            self.b3 = nn.Sequential(
                MyBlock((64, 128, 128), 64, 128),
                MyBlock((128, 128, 128), 128, 128),
                MyBlock((128, 128, 128), 128, 128)
            )
            self.down3 = nn.Conv2d(128, 128, 4, 2, 1) # 64x64 --> 32x32

            self.te4 = self._make_te(time_emb_dim, 128)
            self.b4 = nn.Sequential(
                MyBlock((128, 64, 64), 128, 256),
                MyBlock((256, 64, 64), 256, 256),
                MyBlock((256, 64, 64), 256, 256)
            )
            self.down4 = nn.Conv2d(256, 256, 4, 2, 1) # 32x32 --> 16x16
            
            self.te5 = self._make_te(time_emb_dim, 256)
            self.b5 = nn.Sequential(
                MyBlock((256, 32, 32), 256, 512),
                MyBlock((512, 32, 32), 512, 512),
                MyBlock((512, 32, 32), 512, 512)
            )
            self.down5 = nn.Conv2d(512, 512, 4, 2, 1) # 16x16 --> 8x8

            self.te6 = self._make_te(time_emb_dim, 512)
            self.b6 = nn.Sequential(
                MyBlock((512, 16, 16), 512, 1024),
                MyBlock((1024, 16, 16), 1024, 1024),
                MyBlock((1024, 16, 16), 1024, 1024)
            )
            self.down6 = nn.Conv2d(1024, 1024, 4, 2, 1) # 16x16 --> 8x8

            self.te7 = self._make_te(time_emb_dim, 1024)
            self.b7 = nn.Sequential(
                MyBlock((1024, 8, 8), 1024, 2048),
                MyBlock((2048, 8, 8), 2048, 2048),
                MyBlock((2048, 8, 8), 2048, 2048)
            )            
            self.down7 = nn.Sequential(
                nn.Conv2d(2048, 2048, 2, 1), # 8x8 --> 7x7
                nn.GELU(),
                nn.Conv2d(2048, 2048, 4, 2, 2) # 7x7 --> 4x4
            )

            # Bottleneck
            self.te_mid = self._make_te(time_emb_dim, 2048)
            self.b_mid = nn.Sequential(
                MyBlock((2048, 4, 4), 2048, 2048),
                MyBlock((2048, 4, 4), 2048, 2048),
                MyBlock((2048, 4, 4), 2048, 2048)
            )

            # Second half
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(2048, 2048, 3, 2, 1), #4x4 --> 7x7
                nn.GELU(),
                nn.ConvTranspose2d(2048, 2048, 2, 1) #7x7 --> 8x8
            )

            self.te8 = self._make_te(time_emb_dim, 4096) # 2048 (out7) + 2048 = 4096
            self.b8 = nn.Sequential(
                MyBlock((4096, 8, 8), 4096, 2048),
                MyBlock((2048, 8, 8), 2048, 1024),
                MyBlock((1024, 8, 8), 1024, 1024)
            )

            self.up2 = nn.ConvTranspose2d(1024, 1024, 4, 2, 1) #8x8 --> 16x16
            self.te9 = self._make_te(time_emb_dim, 2048) # 1024 (out6) + 1024 = 2048
            self.b9 = nn.Sequential(
                MyBlock((2048, 16, 16), 2048, 1024),
                MyBlock((1024, 16, 16), 1024, 512),
                MyBlock((512, 16, 16), 512, 512)
            )

            self.up3 = nn.ConvTranspose2d(512, 512, 4, 2, 1) #16x16 --> 32x32
            self.te10 = self._make_te(time_emb_dim, 1024) # 512 (out5) + 512 = 1024
            self.b10 = nn.Sequential(
                MyBlock((1024, 32, 32), 1024, 512),
                MyBlock((512, 32, 32), 512, 256),
                MyBlock((256, 32, 32), 256, 256, normalize=False)
            )

            self.up4 = nn.ConvTranspose2d(256, 256, 4, 2, 1) #32x32 --> 64x64
            self.te11 = self._make_te(time_emb_dim, 512) # 256 (out4) + 256 = 512
            self.b11 = nn.Sequential(
                MyBlock((512, 64, 64), 512, 256),
                MyBlock((256, 64, 64), 256, 128),
                MyBlock((128, 64, 64), 128, 128, normalize=False)
            )

            self.up5 = nn.ConvTranspose2d(128, 128, 4, 2, 1) #64x64 --> 128x128
            self.te12 = self._make_te(time_emb_dim, 256) # 128 (out3) + 128 = 256
            self.b12 = nn.Sequential(
                MyBlock((256, 128, 128), 256, 128),
                MyBlock((128, 128, 128), 128, 64),
                MyBlock((64, 128, 128), 64, 64, normalize=False)
            )
            
            self.up6 = nn.ConvTranspose2d(64, 64, 4, 2, 1) #64x64 --> 128x128
            self.te13 = self._make_te(time_emb_dim, 128) # 64 (out2) + 64 = 128
            self.b13 = nn.Sequential(
                MyBlock((128, 256, 256), 128, 64),
                MyBlock((64, 256, 256), 64, 32),
                MyBlock((32, 256, 256), 32, 32, normalize=False)
            )
            
            self.up7 = nn.ConvTranspose2d(32, 32, 4, 2, 1) #64x64 --> 128x128
            self.te14 = self._make_te(time_emb_dim, 64) # 32 (out1) + 32 = 64
            self.b14 = nn.Sequential(
                MyBlock((64, 512, 512), 64, 32),
                MyBlock((32, 512, 512), 32, 16),
                MyBlock((16, 512, 512), 16, 16, normalize=False)
            )

            self.conv_out = nn.Conv2d(16, channel, 3, 1, 1)
            
        else: 
            print(f"Wrong size : {self.size}")

    def forward(self, x, t):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            t (torch.Tensor): The time tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        t = self.time_embed(t)
        n = len(x)
        if self.size == 128:
            out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
            out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
            out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))
            out4 = self.b4(self.down3(out3) + self.te4(t).reshape(n, -1, 1, 1))
            out5 = self.b5(self.down4(out4) + self.te5(t).reshape(n, -1, 1, 1))
            
            out_mid = self.b_mid(self.down5(out5) + self.te_mid(t).reshape(n, -1, 1, 1))

            out6 = torch.cat((out5, self.up1(out_mid)), dim=1)
            out6 = self.b6(out6 + self.te6(t).reshape(n, -1, 1, 1))

            out7 = torch.cat((out4, self.up2(out6)), dim=1)
            out7 = self.b7(out7 + self.te7(t).reshape(n, -1, 1, 1))

            out8 = torch.cat((out3, self.up3(out7)), dim=1)
            out8 = self.b8(out8 + self.te8(t).reshape(n, -1, 1, 1))

            out9 = torch.cat((out2, self.up4(out8)), dim=1)
            out9 = self.b9(out9 + self.te9(t).reshape(n, -1, 1, 1))

            out10 = torch.cat((out1, self.up5(out9)), dim=1)
            out = self.b10(out10 + self.te10(t).reshape(n, -1, 1, 1))

            out = self.conv_out(out)

            return out
        
        elif self.size == 256:
            out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))        
            out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))        
            out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))        
            out4 = self.b4(self.down3(out3) + self.te4(t).reshape(n, -1, 1, 1))        
            out5 = self.b5(self.down4(out4) + self.te5(t).reshape(n, -1, 1, 1))
            out6 = self.b6(self.down5(out5) + self.te6(t).reshape(n, -1, 1, 1))
            
            out_mid = self.b_mid(self.down6(out6) + self.te_mid(t).reshape(n, -1, 1, 1))

            out7 = torch.cat((out6, self.up1(out_mid)), dim=1)
            out7 = self.b7(out7 + self.te7(t).reshape(n, -1, 1, 1))

            out8 = torch.cat((out5, self.up2(out7)), dim=1)
            out8 = self.b8(out8 + self.te8(t).reshape(n, -1, 1, 1))

            out9 = torch.cat((out4, self.up3(out8)), dim=1)
            out9 = self.b9(out9 + self.te9(t).reshape(n, -1, 1, 1))

            out10 = torch.cat((out3, self.up4(out9)), dim=1)
            out10 = self.b10(out10 + self.te10(t).reshape(n, -1, 1, 1))

            out11 = torch.cat((out2, self.up5(out10)), dim=1)
            out11 = self.b11(out11 + self.te11(t).reshape(n, -1, 1, 1))
            
            out12 = torch.cat((out1, self.up6(out11)), dim=1)
            out = self.b12(out12 + self.te12(t).reshape(n, -1, 1, 1))

            out = self.conv_out(out)

            return out
        
        elif self.size == 512:
            out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))        
            out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))        
            out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))        
            out4 = self.b4(self.down3(out3) + self.te4(t).reshape(n, -1, 1, 1))        
            out5 = self.b5(self.down4(out4) + self.te5(t).reshape(n, -1, 1, 1))
            out6 = self.b6(self.down5(out5) + self.te6(t).reshape(n, -1, 1, 1))
            out7 = self.b7(self.down6(out6) + self.te7(t).reshape(n, -1, 1, 1))
            
            out_mid = self.b_mid(self.down7(out7) + self.te_mid(t).reshape(n, -1, 1, 1))

            out8 = torch.cat((out7, self.up1(out_mid)), dim=1)
            out8 = self.b8(out8 + self.te8(t).reshape(n, -1, 1, 1))

            out9 = torch.cat((out6, self.up2(out8)), dim=1)
            out9 = self.b9(out9 + self.te9(t).reshape(n, -1, 1, 1))

            out10 = torch.cat((out5, self.up3(out9)), dim=1)
            out10 = self.b10(out10 + self.te10(t).reshape(n, -1, 1, 1))

            out11 = torch.cat((out4, self.up4(out10)), dim=1)
            out11 = self.b11(out11 + self.te11(t).reshape(n, -1, 1, 1))

            out12 = torch.cat((out3, self.up5(out11)), dim=1)
            out12 = self.b12(out12 + self.te12(t).reshape(n, -1, 1, 1))
            
            out13 = torch.cat((out2, self.up6(out12)), dim=1)
            out13 = self.b13(out13 + self.te13(t).reshape(n, -1, 1, 1))
            
            out14 = torch.cat((out1, self.up7(out13)), dim=1)
            out = self.b14(out14 + self.te14(t).reshape(n, -1, 1, 1))

            out = self.conv_out(out)

            return out
            

    def _make_te(self, dim_in, dim_out):
        """
        Creates a sequential neural network model with two linear layers and a SiLU activation function.

        Args:
            dim_in (int): The number of input features.
            dim_out (int): The number of output features.

        Returns:
            torch.nn.Sequential: The sequential neural network model.
        """
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.Linear(dim_out, dim_out)
        )