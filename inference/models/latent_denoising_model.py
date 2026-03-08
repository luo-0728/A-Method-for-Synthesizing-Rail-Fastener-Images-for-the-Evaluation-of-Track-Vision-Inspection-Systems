import logging
from collections import OrderedDict
import os
import numpy as np
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm

from ema_pytorch import EMA

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss

from .base_model import BaseModel

logger = logging.getLogger("base")


class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        os.makedirs('image', exist_ok=True)

        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt["train"]

        # define network and load pretrained models
        self.model = networks.define_G(opt).to(self.device)
        self.latent_model = networks.define_L(opt).to(self.device)

        self.recons_model = networks.define_G_recons(opt).to(self.device)

        for param in self.latent_model.parameters():
                param.requires_grad = False

        for param in self.recons_model.parameters():
                param.requires_grad = False

        if opt["dist"]:
            self.model = DistributedDataParallel(self.model, device_ids=[torch.cuda.current_device()])

        self.load()

        self.encode = self.latent_model.encode
        self.decode = self.latent_model.decode

        if self.is_train:
            self.model.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.weight = opt['train']['weight']
            self.weight_jubu = opt['train']['weight_jubu']
            self.weight_feature = opt['train']['weight_feature']
            self.weight_noise = opt['train']['weight_noise']
            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (
                k,
                v,
            ) in self.model.named_parameters():  # can optimize for a part of the model
                #if 'NAFBlock' in k:
                #    v.requires_grad = False
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))

            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )

            self.optimizers.append(self.optimizer)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt["T_period"],
                            eta_min=train_opt["eta_min"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

    def feed_data(self, state, state_recons, LQ, GT=None, MA=None):
        self.state = state.to(self.device)    # noisy_state
        self.condition = LQ.to(self.device)  # LQ
        if state_recons!=None:
            self.state_recons = state_recons.to(self.device)
        if GT is not None:
            self.state_0 = GT.to(self.device)  # GT
        else:
            self.state_0 = None
        if MA is not None:
            #print(1)
            self.MA = MA.to(self.device)  # GT
        else:
            #print(2)
            self.MA = None
        

    def optimize_parameters(self, step, timesteps, sde=None):
        sde.set_mu(self.condition, self.MA,  self.state_0)

        self.optimizer.zero_grad()

        timesteps = timesteps.to(self.device)

        # Get noise and score
        noise_recons, encs_recons, decs_recons = sde.noise_fn_recons(self.state, timesteps.squeeze())
        noise,encs, decs = sde.noise_fn(self.state, timesteps.squeeze())
        
        score = sde.get_score_from_noise(noise, timesteps)
        criterion = nn.L1Loss()
        loss_noise = criterion(noise, noise_recons)
        #criterion2 = nn.MSELoss()
        loss_recons_ = 0.0
        for i, (feat1, feat2) in enumerate(zip(decs, decs_recons)):
            # 维度验证
            assert feat1.shape == feat2.shape, \
                f"第 {i} 层形状不匹配: {feat1.shape} vs {feat2.shape}"
            feat1_norm = F.normalize(feat1, dim=1)  # 对 channel 维度归一化
            feat2_norm = F.normalize(feat2, dim=1)
            layer_loss = criterion(feat1_norm, feat2_norm).mean()
            weight = (i + 1) / len(decs)
            loss_recons_ += weight * layer_loss



        # loss_recons_decs_ = 0.0
        # for i, (feat1, feat2) in enumerate(zip(decs, decs_recons)):
        #     # 维度验证
        #     assert feat1.shape == feat2.shape, \
        #         f"第 {i} 层形状不匹配: {feat1.shape} vs {feat2.shape}"
        #     layer_loss = criterion(feat1, feat2).mean()
        #     loss_recons_decs_ += layer_loss
        # Learning the maximum likelihood objective for state x_{t-1}
        xt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)#实际的xt-1
        xt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)#理想的xt-1
        loss_whole = self.loss_fn(xt_1_expection, xt_1_optimum)
        loss_jubu = self.loss_fn(xt_1_expection * self.MA, xt_1_optimum * self.MA)
        loss = self.weight * self.loss_fn(xt_1_expection, xt_1_optimum) + self.weight_jubu * self.loss_fn(xt_1_expection * self.MA, xt_1_optimum * self.MA) + self.weight_feature*loss_recons_ + self.weight_noise*loss_noise
        # 

        loss.backward()
        self.optimizer.step()
        self.ema.update()

        # set log
        self.log_dict["loss_whole"] = self.weight * loss_whole.item()
        self.log_dict["loss_jubu"] = self.weight_jubu * loss_jubu.item()
        self.log_dict["loss_recons_"] = self.weight_feature*loss_recons_.item()
        self.log_dict["loss_noise"] = self.weight_noise*loss_noise.item()
        self.log_dict["loss"] = loss.item()

    def test(self, sde=None, hidden=None, perform_ode=False, save_states=False):
        #print(self.MA.shape)
        GT = None
        sde.set_mu(self.condition, self.MA, GT)

        self.model.eval()
        with torch.no_grad():
            if not perform_ode:
                # for SDE
                latent = sde.reverse_sde(self.state, save_states=save_states)
            else:
                # if perform Denoising ODE
                latent = sde.reverse_ode(self.state, save_states=save_states)

            self.output = self.decode(latent, hidden)
            # errors = latent - self.state_0  # 形状：(n_samples, 2)

            # # 误差平方
            # squared_errors = errors ** 2
            # squared_errors = squared_errors.detach().cpu().numpy()
            # mse = np.mean(squared_errors)
            # print(f"MSE of two features: {mse:.4f}")
            #latent_mse = torch.mean((latent - self.state_0).pow(2)).item()
            #print(f"[Latent MSE] Recovery error in latent space: {latent_mse:.6f}")
        self.model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])

        load_path_L = self.opt["path"]["pretrain_model_L"]
        if load_path_L is not None:
            logger.info("Loading model for L [{:s}] ...".format(load_path_L))
            self.load_network(load_path_L, self.latent_model, self.opt["path"]["strict_load"])

        load_path_G_recons = self.opt["path"]["pretrain_model_G_recons"]
        if load_path_G_recons is not None:
            logger.info("Loading model for G_recons [{:s}] ...".format(load_path_G_recons))
            self.load_network(load_path_G_recons, self.recons_model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')

