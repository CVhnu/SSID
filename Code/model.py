"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from gcl.models.gan import AdaINGen, MsImageDis, VGGLoss
from gcl.utils.gan_utils import get_model_list, get_scheduler
import torch
import torch.nn as nn
import os


######################################################################
# Load model


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)


class DehazeNet(nn.Module):
    def __init__(self, hyperparameters):
        super(DehazeNet, self).__init__()
        lr_g = hyperparameters['lr_g']
        lr_d = hyperparameters['lr_d']
        # Initiate the networks
        self.gen = AdaINGen(hyperparameters['input_dim'], hyperparameters['gen'],
                            fp16=False)
        self.dis = MsImageDis(3, hyperparameters['dis'], fp16=False)

        self.VggLoss = VGGLoss()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis.parameters())  # + list(self.dis_b.parameters())
        gen_params = list(self.gen.parameters())  # + list(self.gen_b.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr_d, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr_g, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

    def recon_criterion(self, input, target):
        diff = input - target.detach()
        return torch.mean(torch.abs(diff[:]))


    def forward(self, clear, pseudo_hazy, hyperparameters):
        # encode
        s_pseudo_hazy = self.gen.encode(pseudo_hazy)

        # decode
        x_recon = self.gen.decode(s_pseudo_hazy)

        # concat clear and output_decode
        # test = torch.cat((clear, x_recon), 1)
        x_attion = self.gen.coordattention(torch.cat((pseudo_hazy, x_recon), 1))

        x_dehazed = self.gen.refine(x_attion)

        # D loss
        loss_dis_recon, _ = self.dis.calc_dis_loss(self.dis, x_dehazed.detach(), clear)

        loss_dis_total = hyperparameters['gan_w'] * loss_dis_recon

        # GAN loss
        # adv loss
        loss_gen_adv_recon = self.dis.calc_gen_loss(self.dis, x_dehazed.detach())
        # auto-encoder image reconstruction
        loss_gen_recon_x = self.recon_criterion(x_dehazed, clear)

        # feature reconstruction
        loss_gen_recon_f = self.VggLoss(x_dehazed, clear)

        loss_gen_total = hyperparameters['gan_w'] * loss_gen_adv_recon + \
                              hyperparameters['recon_x_w'] * loss_gen_recon_x + \
                              hyperparameters['recon_f_w'] * loss_gen_recon_f
        return x_dehazed, loss_dis_total, loss_gen_total, loss_gen_adv_recon, loss_gen_recon_x, loss_gen_recon_f


    def gen_update(self, loss_g):
        self.gen_opt.zero_grad()
        self.loss_gen = loss_g
        self.loss_gen.backward()
        self.gen_opt.step()

    def dis_update(self, loss_d):
        self.dis_opt.zero_grad()
        self.loss_dis = loss_d
        self.loss_dis.backward()
        self.dis_opt.step()

    def sample(self, pseudo_hazy):
        self.eval()
        # encode
        s_pseudo_hazy = self.gen.encode(pseudo_hazy)

        # decode
        x_recon = self.gen.decode(s_pseudo_hazy)

        # concat clear and output_decode
        x_attion = self.gen.coordattention(torch.cat((pseudo_hazy, x_recon), 1))

        x_dehazed = self.gen.refine(x_attion)

        return x_dehazed

    def sample_recon(self, x_img, x_mesh):
        self.eval()
        # encode
        s_org = self.gen.encode(x_mesh)
        feat = self.id_net(x_img, mode='display')

        # decode
        x_recon = self.gen.decode(s_org, feat)

        return x_recon

    def sample_nv(self, x_img, x_mesh_nv):
        self.eval()
        # encode
        s_nv = self.gen.encode(x_mesh_nv)
        feat = self.id_net(x_img, mode='display')

        # decode
        x_nv = self.gen.decode(s_nv, feat)
        return x_nv

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict['gen'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict['dis'])
        # Load optimizers
        try:
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.dis_opt.load_state_dict(state_dict['dis'])
            self.gen_opt.load_state_dict(state_dict['gen'])
        except:
            pass
        self.dis_opt.param_groups[0]['lr'] = 0.0001
        self.gen_opt.param_groups[0]['lr'] = 0.0001
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'gen': self.gen.state_dict()}, gen_name)
        torch.save({'dis': self.dis.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

