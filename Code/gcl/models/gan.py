"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import math
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torchvision import models
import torch.nn.init as init
from torchvision.models import vgg16

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


##################################################################################
# Discriminator
##################################################################################

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params, fp16):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.LAMBDA = params['LAMBDA']
        self.non_local = params['non_local']
        self.n_res = params['n_res']
        self.input_dim = input_dim
        self.fp16 = fp16
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        if not self.gan_type == 'wgan':
            self.cnns = nn.ModuleList()
            for _ in range(self.num_scales):
                Dis = self._make_net()
                Dis.apply(weights_init('gaussian'))
                self.cnns.append(Dis)
        else:
            self.cnn = self.one_cnn()

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [
            Conv2dBlock(self.input_dim, dim, 1, 1, 0, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
        cnn_x += [Conv2dBlock(dim, dim, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
        cnn_x += [Conv2dBlock(dim, dim, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            dim2 = min(dim * 2, 512)
            cnn_x += [Conv2dBlock(dim, dim, 3, 1, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            cnn_x += [Conv2dBlock(dim, dim2, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim = dim2
        if self.non_local > 1:
            cnn_x += [NonlocalBlock(dim)]
        for i in range(self.n_res):
            cnn_x += [ResBlock(dim, norm=self.norm, activation=self.activ, pad_type=self.pad_type, res_type='basic')]
        if self.non_local > 0:
            cnn_x += [NonlocalBlock(dim)]
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def one_cnn(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(5):
            dim2 = min(dim * 2, 512)
            cnn_x += [Conv2dBlock(dim, dim2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim = dim2
        cnn_x += [nn.Conv2d(dim, 1, (4, 2), 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        if not self.gan_type == 'wgan':
            outputs = []
            for model in self.cnns:
                outputs.append(model(x))
                x = self.downsample(x)
        else:
            outputs = self.cnn(x)
            outputs = torch.squeeze(outputs)
        return outputs

    def calc_dis_loss(self, model, input_fake, input_real):
        # calculate the loss to train D
        input_real.requires_grad_()
        outs0 = model.forward(input_fake)
        outs1 = model.forward(input_real)
        loss = 0
        reg = 0
        Drift = 0.001
        LAMBDA = self.LAMBDA

        if self.gan_type == 'wgan':
            loss += torch.mean(outs0) - torch.mean(outs1)
            loss += Drift * (torch.sum(outs0 ** 2) + torch.sum(outs1 ** 2))
            reg += LAMBDA * self.compute_grad2(outs1, input_real).mean()  # I suggest Lambda=0.1 for wgan
            loss = loss + reg
            return loss, reg

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
                # regularization
                reg += LAMBDA * self.compute_grad2(out1, input_real).mean()
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
                reg += LAMBDA * self.compute_grad2(F.sigmoid(out1), input_real).mean()
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        loss = loss + reg
        return loss, reg

    def calc_gen_loss(self, model, input_fake):
        # calculate the loss to train G
        outs0 = model.forward(input_fake)
        loss = 0
        Drift = 0.001
        if self.gan_type == 'wgan':
            loss += -torch.mean(outs0)
            # progressive gan
            loss += Drift * torch.sum(outs0 ** 2)
            return loss

        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1) ** 2) * 2  # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg

##################################################################################
# Attention
##################################################################################


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # self.pool_h = nn.AdaptiveAvgPool2d((512, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, 512))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        # self.bn1 = nn.InstanceNorm2d(mip)  #cjy

        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

##################################################################################
# Generator
##################################################################################

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params, fp16):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        which_dec = params['dec']
        dropout = params['dropout']
        tanh = params['tanh']
        non_local = params['non_local']

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type,
                                          dropout=dropout, tanh=tanh, res_type='basic')

        self.output_dim = self.enc_content.output_dim

        self.dec = Decoder(n_downsample, n_res, self.output_dim, 3, dropout=dropout, res_norm='in', activ=activ,
                               pad_type=pad_type, res_type='basic', non_local=non_local, fp16=fp16)

        self.catt = CoordAtt(input_dim*2, input_dim*2)

        self.ref = Refine(input_dim*2, input_dim)


    def encode(self, images):
        # encode an image to its content and style codes
        content = self.enc_content(images)
        return content

    def decode(self, content):
        images = self.dec(content)
        return images

    def coordattention(self, content):
        images = self.catt(content)
        return images

    def refine(self, content):
        images = self.ref(content)
        return images


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type, dropout, tanh=False,
                 res_type='basic'):
        super(ContentEncoder, self).__init__()
        self.model = []
        # Here I change the stride to 2.
        self.model += [Conv2dBlock(input_dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, 2 * dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        dim *= 2  # 32dim
        # downsampling blocks
        for i in range(n_downsample - 1):
            self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
            self.model += [Conv2dBlock(dim, 2 * dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type, res_type=res_type)]
        # 64 -> 128
        # self.model += [ASPP(dim, norm=norm, activation=activ, pad_type=pad_type)]
        # dim *= 2
        if tanh:
            self.model += [nn.Tanh()]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Refine(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Refine, self).__init__()
        self.refine = []
        self.refine += [nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)]
        self.refine += [nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)]
        # self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        # self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        # self.model += [Conv2dBlock(dim, output_dim, 1, 1, 0, norm='none', activation='none', pad_type=pad_type)]
        self.refine = nn.Sequential(*self.refine)

    def forward(self, x):
        output = self.refine(x)
        return output


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, dropout=0, res_norm='in', activ='relu', pad_type='zero',
                 res_type='basic', non_local=False, fp16=False):
        super(Decoder, self).__init__()
        self.input_dim = dim
        self.model = []
        self.model += [nn.Dropout(p=dropout)]
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type, res_type=res_type)]
        # non-local
        if non_local > 0:
            self.model += [NonlocalBlock(dim)]
            print('use non-local!')
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type,
                                       fp16=fp16)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, output_dim, 1, 1, 0, norm='none', activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        output = self.model(x)
        return output


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', res_type='basic'):
        super(ResBlocks, self).__init__()
        self.model = []
        self.res_type = res_type
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, res_type=res_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm, activation='relu', pad_type='zero', res_type='basic'):
        super(ResBlock, self).__init__()

        model = []
        if res_type == 'basic' or res_type == 'nonlocal':
            model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        elif res_type == 'slim':
            dim_half = dim // 2
            model += [Conv2dBlock(dim, dim_half, 1, 1, 0, norm='in', activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim_half, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Conv2dBlock(dim_half, dim, 1, 1, 0, norm='in', activation='none', pad_type=pad_type)]
        elif res_type == 'series':
            model += [Series2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Series2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        elif res_type == 'parallel':
            model += [Parallel2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
            model += [Parallel2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        else:
            print('unkown block type')
        self.res_type = res_type
        self.model = nn.Sequential(*model)
        if res_type == 'nonlocal':
            self.nonloc = NonlocalBlock(dim)

    def forward(self, x):
        if self.res_type == 'nonlocal':
            x = self.nonloc(x)
        residual = x
        out = self.model(x)
        out += residual
        return out


class NonlocalBlock(nn.Module):
    def __init__(self, in_dim, norm='in'):
        super(NonlocalBlock, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class ASPP(nn.Module):
    # ASPP (a)
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ASPP, self).__init__()
        dim_part = dim // 2
        self.conv1 = Conv2dBlock(dim, dim_part, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type)

        self.conv6 = []
        self.conv6 += [Conv2dBlock(dim, dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv6 += [
            Conv2dBlock(dim_part, dim_part, 3, 1, 3, norm=norm, activation='none', pad_type=pad_type, dilation=3)]
        self.conv6 = nn.Sequential(*self.conv6)

        self.conv12 = []
        self.conv12 += [Conv2dBlock(dim, dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv12 += [
            Conv2dBlock(dim_part, dim_part, 3, 1, 6, norm=norm, activation='none', pad_type=pad_type, dilation=6)]
        self.conv12 = nn.Sequential(*self.conv12)

        self.conv18 = []
        self.conv18 += [Conv2dBlock(dim, dim_part, 1, 1, 0, norm=norm, activation=activation, pad_type=pad_type)]
        self.conv18 += [
            Conv2dBlock(dim_part, dim_part, 3, 1, 9, norm=norm, activation='none', pad_type=pad_type, dilation=9)]
        self.conv18 = nn.Sequential(*self.conv18)

        self.fuse = Conv2dBlock(4 * dim_part, 2 * dim, 1, 1, 0, norm=norm, activation='none', pad_type=pad_type)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv6 = self.conv6(x)
        conv12 = self.conv12(x)
        conv18 = self.conv18(x)
        out = torch.cat((conv1, conv6, conv12, conv18), dim=1)
        out = self.fuse(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1, fp16=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim, fp16=fp16)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = torch.nn.utils.spectral_norm(
                nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class Series2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Series2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

        self.instance_norm = nn.InstanceNorm2d(norm_dim)

    def forward(self, x):
        x = self.conv(self.pad(x))
        x = self.norm(x) + x
        x = self.instance_norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class Parallel2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Parallel2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

        self.instance_norm = nn.InstanceNorm2d(norm_dim)

    def forward(self, x):
        x = self.conv(self.pad(x)) + self.norm(x)
        x = self.instance_norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            # reshape input
            out = out.unsqueeze(1)
            out = self.norm(out)
            out = out.view(out.size(0), out.size(2))
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VggLoss(nn.Module):
    def __init__(self, feat=[2, 5, 9, 13]):
        super(VggLoss, self).__init__()

        net = vgg16(pretrained=True)
        self.vgg16_ac = VggActivations(net, feat)  # [2, 5, 9, 13, 17])

    def forward(self, x_pred, x_real):
        y_feat_real = self.vgg16_ac(x_real)
        y_feat_pred = self.vgg16_ac(x_pred)

        loss = 0
        for i in range(len(y_feat_real)):
            loss += (y_feat_real[i] - y_feat_pred[i]).abs().mean()

        return loss


class VggActivations(nn.Module):
    def __init__(self, vgg_network, feature_idx):
        super(VggActivations, self).__init__()

        features = list(vgg_network.features)
        self.features = nn.ModuleList(features).eval()
        self.idx_list = feature_idx

    def forward(self, x):

        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.idx_list:
                results.append(x)

        return results


##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b).type_as(x)
        running_var = self.running_var.repeat(b).type_as(x)
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, fp16=False):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.fp16 = fp16
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.type() == 'torch.cuda.HalfTensor':  # For Safety
            mean = x.view(-1).float().mean().view(*shape)
            std = x.view(-1).float().std().view(*shape)
            mean = mean.half()
            std = std.half()
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
