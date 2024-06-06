import os

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class AdaptiveSine(nn.Module):

    def __init__(self, alpha=1.0, fixed=False):
        super().__init__()
        if fixed:
            self.freq = alpha
        else:
            # learnable frequency
            self.freq = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, x):
        return torch.sin(self.freq * x)


class FourierConv2D(nn.Module):

    def __init__(self, in_c, out_c, wavenumber1, wavenumber2):
        super().__init__()
        self.out_ = out_c
        self.wavenumber1 = wavenumber1
        self.wavenumber2 = wavenumber2
        scale = (1 / (in_c * out_c))
        self.weights1 = nn.Parameter(scale * torch.rand(in_c, out_c, wavenumber1, wavenumber2, 2, dtype=torch.float32))
        self.weights2 = nn.Parameter(scale * torch.rand(in_c, out_c, wavenumber1, wavenumber2, 2, dtype=torch.float32))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ,2), (in_channel, out_channel, x,y,2) -> (batch, out_channel, x,y)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        # input: batch,channel,x,y
        # out: batch,channel,x,y
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.view_as_real(torch.fft.rfft2(x))  # input: batch,channel,x,y->batch,channel,x,y,2
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_,  x.size(-2), x.size(-1)//2 + 1,2, dtype=torch.float32, device=x.device)
        out_ft[:, :, :self.wavenumber1, :self.wavenumber2,:] = \
            self.compl_mul2d(x_ft[:, :, :self.wavenumber1, :self.wavenumber2,:], self.weights1)
        out_ft[:, :, -self.wavenumber1:, :self.wavenumber2,:] = \
            self.compl_mul2d(x_ft[:, :, -self.wavenumber1:, :self.wavenumber2,:], self.weights2)
        # Return to physical space
        x = torch.fft.irfft2(torch.view_as_complex(out_ft), s=(x.size(-2), x.size(-1)))
        return x


class BornFourierConv2D(nn.Module):

    def __init__(self, in_c, out_c, wavenumber1, wavenumber2):
        super().__init__()
        self.out_ = out_c
        self.wavenumber1 = wavenumber1
        self.wavenumber2 = wavenumber2
        scale = (1 / (in_c * out_c))
        self.weights1 = nn.Parameter(scale * torch.rand(in_c, out_c, wavenumber1, wavenumber2, 2, dtype=torch.float32))
        self.weights2 = nn.Parameter(scale * torch.rand(in_c, out_c, wavenumber1, wavenumber2, 2, dtype=torch.float32))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ,2), (in_channel, out_channel, x,y,2) -> (batch, out_channel, x,y)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, x_eps):
        # input: batch,channel,x,y
        # out: batch,channel,x,y
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.view_as_real(torch.fft.rfft2(x * x_eps))  # input: batch,channel,x,y->batch,channel,x,y,2
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_,  x.size(-2), x.size(-1)//2 + 1,2, dtype=torch.float32, device=x.device)
        out_ft[:, :, :self.wavenumber1, :self.wavenumber2,:] = \
            self.compl_mul2d(x_ft[:, :, :self.wavenumber1, :self.wavenumber2,:], self.weights1)
        out_ft[:, :, -self.wavenumber1:, :self.wavenumber2,:] = \
            self.compl_mul2d(x_ft[:, :, -self.wavenumber1:, :self.wavenumber2,:], self.weights2)
        # Return to physical space
        x = torch.fft.irfft2(torch.view_as_complex(out_ft), s=(x.size(-2), x.size(-1)))
        return x


class SimplifiedBornFourierConv2D(nn.Module):

    def __init__(self, in_c, out_c, wavenumber1, wavenumber2):
        super().__init__()
        self.out_ = out_c
        self.wavenumber1 = wavenumber1
        self.wavenumber2 = wavenumber2
        scale = (1 / (in_c * out_c))
        self.weights1 = nn.Parameter(scale * torch.rand(in_c, out_c, wavenumber1, wavenumber2, dtype=torch.float32))
        self.weights2 = nn.Parameter(scale * torch.rand(in_c, out_c, wavenumber1, wavenumber2, dtype=torch.float32))

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ,2), (in_channel, out_channel, x,y,2) -> (batch, out_channel, x,y)
        return torch.einsum("bixy...,ioxy->boxy...", input, weights)

    def forward(self, x, x_eps):
        # input: batch,channel,x,y
        # out: batch,channel,x,y
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.view_as_real(torch.fft.rfft2(x * x_eps))  # input: batch,channel,x,y->batch,channel,x,y,2
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_,  x.size(-2), x.size(-1)//2 + 1,2, dtype=torch.float32, device=x.device)
        out_ft[:, :, :self.wavenumber1, :self.wavenumber2,:] = \
            self.compl_mul2d(x_ft[:, :, :self.wavenumber1, :self.wavenumber2,:], self.weights1)
        out_ft[:, :, -self.wavenumber1:, :self.wavenumber2,:] = \
            self.compl_mul2d(x_ft[:, :, -self.wavenumber1:, :self.wavenumber2,:], self.weights2)
        # Return to physical space
        x = torch.fft.irfft2(torch.view_as_complex(out_ft), s=(x.size(-2), x.size(-1)))
        return x


class FourierLayer(nn.Module):

    def __init__(self,  features_, wavenumber, activation='relu', is_last=False):
        super().__init__()
        self.W = nn.Conv2d(features_, features_, 1)
        self.fourier_conv = FourierConv2D(features_, features_ , *wavenumber)
        if is_last:
            self.act = nn.Identity()
        else:
            if activation == 'relu':
                self.act = F.relu
            elif activation == 'gelu':
                self.act = F.gelu
            elif activation == 'sine':
                self.act = AdaptiveSine(alpha=1.0, fixed=True)
            elif activation == 'adaptive_sine':
                self.act = AdaptiveSine(alpha=1.0, fixed=False)
            else:
                raise NotImplementedError(f'Activation {activation} not implemented')

    def forward(self, x):
        x1 = self.fourier_conv(x)
        x2 = self.W(x)
        return self.act(x1 + x2)


class BornFourierLayer(nn.Module):

    def __init__(self,  features_, wavenumber, activation='relu', is_last=False, is_bn=False, simplified_fourier=False):
        super().__init__()
        # self.W = nn.Sequential(
        #     nn.Conv2d(features_, features_ * 2, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(features_ * 2) if is_bn else nn.Identity(),
        #     nn.GELU(),
        #     nn.Conv2d(features_ * 2, features_, kernel_size=3, padding=1),
        # )
        self.W = nn.Conv2d(features_, features_, 1)
        if is_bn:
            self.bn = nn.BatchNorm2d(features_)
        else:
            self.bn = nn.Identity()
        if simplified_fourier:
            self.fourier_conv = SimplifiedBornFourierConv2D(features_, features_ , *wavenumber)
        else:
            self.fourier_conv = BornFourierConv2D(features_, features_ , *wavenumber)
        if is_last:
            self.act = nn.Identity()
        else:
            if activation == 'relu':
                self.act = F.relu
            elif activation == 'gelu':
                self.act = F.gelu
            elif activation == 'sine':
                self.act = AdaptiveSine(alpha=1.0, fixed=True)
            elif activation == 'adaptive_sine':
                self.act = AdaptiveSine(alpha=1.0, fixed=False)
            else:
                raise NotImplementedError(f'Activation {activation} not implemented')

    def forward(self, x, x_eps):
        x1 = self.fourier_conv(x, x_eps)
        x2 = self.W(x)
        return self.act(self.bn(x1 + x2))


class NBSOFourierLayer(nn.Module):

    def __init__(self,  features_, wavenumber, activation='relu', is_last=False):
        super().__init__()
        self.W = nn.Conv2d(features_, features_, 1)
        self.fourier_conv = BornFourierConv2D(features_, features_ , *wavenumber)
        if is_last:
            self.act = nn.Identity()
        else:
            if activation == 'relu':
                self.act = F.relu
            elif activation == 'gelu':
                self.act = F.gelu
            elif activation == 'sine':
                self.act = AdaptiveSine(alpha=1.0, fixed=True)
            elif activation == 'adaptive_sine':
                self.act = AdaptiveSine(alpha=1.0, fixed=False)
            else:
                raise NotImplementedError(f'Activation {activation} not implemented')

    def forward(self, v_c, w_n, u_n):

        w_n_new = self.fourier_conv(w_n, v_c)
        w_n_new = self.W(w_n_new)
        u_n_new = self.act(u_n + w_n_new)
        return self.act(w_n_new), u_n_new


def set_activ(activation):
    if activation is not None:
        activation = activation.lower()
    if activation == 'relu':
        nonlinear = F.relu
    elif activation == "leaky_relu":
        nonlinear = F.leaky_relu
    elif activation == 'tanh':
        nonlinear = F.tanh
    elif activation == 'sine':
        nonlinear = AdaptiveSine(alpha=1.0, fixed=True)
    elif activation == 'adaptive_sine':
        nonlinear = AdaptiveSine(alpha=1.0, fixed=False)
    elif activation == 'gelu':
        nonlinear = F.gelu
    elif activation == 'elu':
        nonlinear = F.elu_
    elif activation == None:
        nonlinear = nn.Identity()
    else:
        raise Exception('The activation is not recognized from the list')
    return nonlinear


class FCLayer(nn.Module):
    """Fully connected layer """
    def __init__(self, in_feature, out_feature,
                        activation = "gelu",
                        is_normalized = True):
        super().__init__()
        self.LinearBlock = nn.Linear(in_feature, out_feature)
        self.act = set_activ(activation)

    def forward(self, x):
        return self.act(self.LinearBlock(x))


class MLP(nn.Module):
    r"""Simple MLP to code lifting and projection"""
    def __init__(self, sizes=(2, 128, 128, 1),
                 activation='relu',
                 outermost_linear=True,
                 outermost_norm=True,
                 drop=0.):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.net = nn.ModuleList([FCLayer(in_feature=m, out_feature=n, activation=activation)
                                  for m, n in zip(sizes[:-2], sizes[1:-1])
                                  ])
        if outermost_linear == True:
            self.net.append(FCLayer(sizes[-2], sizes[-1], activation=None))
        else:
            self.net.append(FCLayer(sizes[-2], sizes[-1], activation=activation))

    def forward(self, x):
        for module in self.net:
            x = module(x)
            x = self.dropout(x)
        return x


class KAN(nn.Module):
    r"""Simple MLP to code lifting and projection"""
    def __init__(self, sizes=(2, 128, 128, 1),
                 outermost_linear=True,
                 drop=0.,
                 gridsize=64):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.net = nn.ModuleList([NaiveFourierKANLayer(inputdim=m, outdim=n, gridsize=gridsize)
                                  for m, n in zip(sizes[:-2], sizes[1:-1])
                                  ])
        if outermost_linear == True:
            self.net.append(NaiveFourierKANLayer(sizes[-2], sizes[-1], gridsize=gridsize))
        else:
            self.net.append(NaiveFourierKANLayer(sizes[-2], sizes[-1], gridsize=gridsize))

    def forward(self, x):
        for module in self.net:
            x = module(x)
            x = self.dropout(x)
        return x


def get_grid2D(shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])

    return torch.cat((gridx, gridy), dim=-1).to(device)


class NaiveFourierKANLayer(torch.nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=True):

        # https://github.com/GistNoesis/FourierKAN/blob/main/fftKAN.py

        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        # then each coordinates of the output is of unit variance
        # independently of the various sizes
        self.fouriercoeffs = torch.nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                                (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if (self.addbias):
            self.bias = torch.nn.Parameter(torch.zeros(1, outdim))

    # x.shape ( ... , indim )
    # out.shape ( ..., outdim)
    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))
        # Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        # This should be fused to avoid materializing memory
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        # We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them
        # y = torch.sum(c * self.fouriercoeffs[0:1], (-2, -1))
        # y += torch.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        # if (self.addbias):
        #     y += self.bias
        # End fuse

        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if (self.addbias):
            y += self.bias
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = th.reshape(c,(1,x.shape[0],x.shape[1],self.gridsize))
        s = th.reshape(s,(1,x.shape[0],x.shape[1],self.gridsize))
        y2 = th.einsum( "dbik,djik->bj", th.concat([c,s],axis=0) ,self.fouriercoeffs )
        if( self.addbias):
            y2 += self.bias
        diff = th.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        y = torch.reshape(y, outshape)
        return y


class FNO(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # lifting
        lifting_size = cfg['lifting_size']
        if self.cfg['lifting'] == 'None':
            self.lifting = MLP(
                sizes=[cfg['input_dim'], lifting_size//2, lifting_size],
                activation='relu',
            )
        else:
            raise NotImplementedError(f'Lifting {self.cfg["lifting"]} not implemented')

        # FNO
        fourier_layers = list()
        n_layers = len(cfg['wavenumber'])
        for l in range(n_layers):
            fourier_layers.append(
                FourierLayer(
                    features_=lifting_size,
                    wavenumber=([cfg['wavenumber'][l]]*2),
                    activation=cfg['activation'], is_last=(l == n_layers-1)))
        self.fno = nn.Sequential(*fourier_layers)

        # proj
        if self.cfg['proj'] == 'None':
            self.proj = MLP(
                sizes=[lifting_size, lifting_size//2, 2],
                activation='relu',
            )
        else:
            raise NotImplementedError(f'Projection {self.cfg["proj"]} not implemented')

    def forward(self, input_data, src_data):
        """
        :param input_data: [B, 480, 480, 1]
        :param src_data: [B, 480, 480, 2]
        :return: [B, 480, 480, 2]
        """

        grid = get_grid2D(input_data.shape, input_data.device)
        _input = torch.cat([input_data, grid, src_data], dim=-1)  # [B, 480, 480, 6]
        field = src_data[..., 1:].clone()  # [B, 480, 480, 2]

        # lifting
        lifting_data = self.lifting(_input)  # [B, 480, 480, embedding_size]
        lifting_data = lifting_data.permute(0, 3, 1, 2).contiguous()  # [B, embedding_size, 480, 480]
        lifting_data = nn.functional.pad(lifting_data, [0, self.cfg['padding'], 0, self.cfg['padding']])

        # FNO
        pred = self.fno(lifting_data)  # [B, embedding_size, 480+padding, 480+padding]

        # de padding
        pred = pred[..., :-self.cfg['padding'], :-self.cfg['padding']].permute(0, 2, 3, 1).contiguous()

        # proj
        pred = self.proj(pred)

        pred = torch.view_as_real(torch.view_as_complex(field.to(pred.device)) * (1 + torch.view_as_complex(pred)))
        return pred


class BornFNO(nn.Module):
    # https://github.com/merlresearch/DeepBornFNO/blob/16ee000167a2b3f1eb50d9a1bb94f9a788adb1c5/forward_model/model.py
    # https://merl.com/publications/docs/TR2023-029.pdf

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # lifting
        lifting_size = cfg['lifting_size']
        if self.cfg['lifting'] == 'None':
            self.input_lifting = MLP(
                sizes=[cfg['input_dim']-1, lifting_size//2, lifting_size],
                activation='gelu',
            )
            self.eps_lifting = MLP(
                sizes=[cfg['input_dim']-3, lifting_size // 2, lifting_size],
                activation='gelu',
            )
        else:
            raise NotImplementedError(f'Lifting {self.cfg["lifting"]} not implemented')

        # FNO
        fourier_layers = list()
        n_layers = len(cfg['wavenumber'])
        for l in range(n_layers):
            fourier_layers.append(
                BornFourierLayer(
                    features_=lifting_size,
                    wavenumber=([cfg['wavenumber'][l]]*2),
                    activation=cfg['activation'], is_last=(l == n_layers-1), is_bn=cfg['use_bn']))
        self.fno = nn.ModuleList(fourier_layers)

        # proj
        if self.cfg['proj'] == 'None':
            self.proj = MLP(
                sizes=[lifting_size, lifting_size//2, 2],
                activation='gelu',
            )
        else:
            raise NotImplementedError(f'Projection {self.cfg["proj"]} not implemented')

    def forward(self, input_data, src_data):
        """
        :param input_data: [B, 480, 480, 1]
        :param src_data: [B, 480, 480, 3]
        :return: [B, 480, 480, 2]
        """

        grid = get_grid2D(input_data.shape, input_data.device)
        x_in = torch.cat([src_data, grid], dim=-1)  # [B, 480, 480, 5]
        x_eps = torch.cat([input_data, grid], dim=-1) # [B, 480, 480, 3]
        field = src_data[..., 1:].clone()  # [B, 480, 480, 2]

        # lifting
        input_lifting_data = self.input_lifting(x_in)  # [B, 480, 480, embedding_size]
        input_lifting_data = input_lifting_data.permute(0, 3, 1, 2).contiguous()  # [B, embedding_size, 480, 480]
        input_lifting_data = nn.functional.pad(input_lifting_data, [0, self.cfg['padding'], 0, self.cfg['padding']])

        eps_lifting_data = self.eps_lifting(x_eps)  # [B, 480, 480, embedding_size]
        eps_lifting_data = eps_lifting_data.permute(0, 3, 1, 2).contiguous()
        eps_lifting_data = nn.functional.pad(eps_lifting_data, [0, self.cfg['padding'], 0, self.cfg['padding']])

        # FNO
        for layer in self.fno:
            input_lifting_data = layer(input_lifting_data, eps_lifting_data)

        # de padding
        pred = input_lifting_data[..., :-self.cfg['padding'], :-self.cfg['padding']].permute(0, 2, 3, 1).contiguous()

        # proj
        pred = self.proj(pred)

        pred = torch.view_as_real(torch.view_as_complex(field.to(pred.device)) * (1 + torch.view_as_complex(pred)))
        return pred


class BornFNOV3(nn.Module):
    # https://github.com/merlresearch/DeepBornFNO/blob/16ee000167a2b3f1eb50d9a1bb94f9a788adb1c5/forward_model/model.py
    # https://merl.com/publications/docs/TR2023-029.pdf

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # lifting
        lifting_size = cfg['lifting_size']
        if self.cfg['lifting'] == 'None':
            self.input_lifting = MLP(
                sizes=[cfg['input_dim']-1, lifting_size//2, lifting_size],
                activation='gelu',
            )
            self.eps_lifting = MLP(
                sizes=[cfg['input_dim']-3, lifting_size // 2, lifting_size],
                activation='gelu',
            )
        elif self.cfg['lifting'] == 'conv':
            self.input_lifting = nn.Sequential(
                nn.Conv2d(cfg['input_dim']-1, lifting_size//2, kernel_size=1),
                nn.BatchNorm2d(lifting_size // 2),
                nn.GELU(),
                nn.Conv2d(lifting_size//2, lifting_size, kernel_size=1),
            )
            self.eps_lifting = nn.Sequential(
                nn.Conv2d(cfg['input_dim']-3, lifting_size//2, kernel_size=1),
                nn.BatchNorm2d(lifting_size // 2),
                nn.GELU(),
                nn.Conv2d(lifting_size//2, lifting_size, kernel_size=1),
            )
        else:
            raise NotImplementedError(f'Lifting {self.cfg["lifting"]} not implemented')

        # FNO
        fourier_layers = list()
        n_layers = len(cfg['wavenumber'])
        for l in range(n_layers):
            fourier_layers.append(
                BornFourierLayer(
                    features_=lifting_size,
                    wavenumber=([cfg['wavenumber'][l]]*2),
                    activation=cfg['activation'], is_last=(l == n_layers-1), is_bn=cfg['use_bn'],
                    simplified_fourier=cfg['simplified_fourier']))
        self.fno = nn.ModuleList(fourier_layers)

        # proj
        if self.cfg['proj'] == 'None':
            self.proj = MLP(
                sizes=[lifting_size, lifting_size//2, 2],
                activation='gelu',
            )
        elif self.cfg['proj'] == 'conv':
            self.proj = nn.Sequential(
                nn.Conv2d(lifting_size, lifting_size//2, kernel_size=3, padding=1),
                nn.BatchNorm2d(lifting_size // 2),
                nn.GELU(),
                nn.Conv2d(lifting_size//2, 2, kernel_size=3, padding=1),
            )
        else:
            raise NotImplementedError(f'Projection {self.cfg["proj"]} not implemented')

        # learned mask
        # self.mask = nn.Parameter(torch.ones(480, 480))
        self.mask = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1)
        )

    def forward(self, input_data, src_data):
        """
        :param input_data: [B, 480, 480, 1]
        :param src_data: [B, 480, 480, 3]
        :return: [B, 480, 480, 2]
        """

        grid = get_grid2D(input_data.shape, input_data.device)
        x_in = torch.cat([src_data, grid], dim=-1)  # [B, 480, 480, 5]
        x_eps = torch.cat([input_data, grid], dim=-1) # [B, 480, 480, 3]
        field = src_data[..., 1:].clone()  # [B, 480, 480, 2]

        # lifting
        if self.cfg['lifting'] == 'conv':
            input_lifting_data = self.input_lifting(x_in.permute(0, 3, 1, 2).contiguous()) # [B, embedding_size, 480, 480]
        elif self.cfg['lifting'] == 'None':
            input_lifting_data = self.input_lifting(x_in)  # [B, 480, 480, embedding_size]
            input_lifting_data = input_lifting_data.permute(0, 3, 1, 2).contiguous()  # [B, embedding_size, 480, 480]
        input_lifting_data = nn.functional.pad(input_lifting_data, [0, self.cfg['padding'], 0, self.cfg['padding']])

        if self.cfg['lifting'] == 'conv':
            eps_lifting_data = self.eps_lifting(x_eps.permute(0, 3, 1, 2).contiguous())
        elif self.cfg['lifting'] == 'None':
            eps_lifting_data = self.eps_lifting(x_eps)  # [B, 480, 480, embedding_size]
            eps_lifting_data = eps_lifting_data.permute(0, 3, 1, 2).contiguous()
        eps_lifting_data = nn.functional.pad(eps_lifting_data, [0, self.cfg['padding'], 0, self.cfg['padding']])

        # FNO
        for layer in self.fno:
            input_lifting_data = layer(input_lifting_data, eps_lifting_data)

        # de padding
        pred = input_lifting_data[..., :-self.cfg['padding'], :-self.cfg['padding']]

        # proj
        if self.cfg['proj'] == 'conv':
            pred = self.proj(pred).permute(0, 2, 3, 1).contiguous()
        elif self.cfg['proj'] == 'None':
            pred = self.proj(pred.permute(0, 2, 3, 1).contiguous())

        # copy mask to batch size
        mask = self.mask(input_data.permute(0, 3, 1, 2)).squeeze(1).contiguous()
        # copy mask to (..., 2)
        mask = mask.unsqueeze(-1).expand_as(pred)

        # pred = torch.view_as_real(torch.view_as_complex(field.to(pred.device)) * (mask + torch.view_as_complex(pred)))
        pred = field.to(pred.device) * mask + pred
        #
        # pred = torch.view_as_real(torch.view_as_complex(field.to(pred.device)) * (1 + torch.view_as_complex(pred)))
        return pred


class BornFNOV2(nn.Module):
    # https://github.com/merlresearch/DeepBornFNO/blob/16ee000167a2b3f1eb50d9a1bb94f9a788adb1c5/forward_model/model.py
    # https://merl.com/publications/docs/TR2023-029.pdf
    # downsample the input data to half size
    # upsample the output data to original size

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.input_data_downsample_module = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm2d(1),
            # nn.GELU(),
        )

        self.src_data_downsample_module = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm2d(3),
            # nn.GELU(),
        )

        self.kernel_model = BornFNO(cfg)

        self.output_data_upsample_module = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2, out_channels=16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
            # nn.BatchNorm2d(2),
            # nn.GELU(),
        )

    def forward(self, input_data, src_data):
        """
        :param input_data: [B, 480, 480, 1]
        :param src_data: [B, 480, 480, 3]
        :return: [B, 480, 480, 2]
        """

        input_data = self.input_data_downsample_module(input_data.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        src_data = self.src_data_downsample_module(src_data.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()

        pred = self.kernel_model(input_data, src_data).permute(0, 3, 1, 2).contiguous()
        pred = self.output_data_upsample_module(pred).permute(0, 2, 3, 1).contiguous()
        return pred


class Model(nn.Module):

    def __init__(self, model_path, normalize=False, normalizer=None):
        super(Model, self).__init__()
        model_conf = {
            'input_dim': 6,
            'lifting_size': 60,
            'lifting': 'None',
            'proj': 'None',
            'wavenumber': [100, 100, 100, 100, 100, 100, 100],
            'padding': 32,
            'activation': 'gelu',
            'use_bn': True,
            'simplified_fourier': True
        }

        self.model = BornFNOV3(model_conf).cuda()
        weight_path = os.path.join(model_path, 'fold_0.ckpt')
        weights = torch.load(weight_path, map_location='cuda:0')['state_dict']
        weights = {k[6:]: v for k, v in weights.items()}
        self.model.load_state_dict(weights)
        torch.save(self.model, f'{model_path}/submission.pt')

        # model_path = os.path.join(model_path, 'submission.pt')
        # self.model = torch.load(model_path).cuda()

        self.model.eval()

    def forward(self, sos, src):

        sos = (1500 / sos - 1) * 30
        src[..., 1:] = src[..., 1:] * 2e-3
        src[..., 0] = src[..., 0] / (2 * torch.pi)

        pred = self.model(sos, src)
        return pred / 2e-3


if __name__ == '__main__':
    model = Model('./outputs/models')
    print(model)
    print("Done.")

    test_input = torch.randn(2, 480, 480, 1).cuda()
    test_src = torch.randn(2, 480, 480, 3).cuda()
    print(model(test_input, test_src).shape)
