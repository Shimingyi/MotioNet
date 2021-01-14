import numpy as np
import torch
import random

from model import metric
from base.base_trainer import base_trainer

from utils import h36m_utils, visualization
from utils.logger import Logger

class fk_trainer(base_trainer):
    def __init__(self, model, resume, config, data_loader, test_data_loader):
        super(fk_trainer, self).__init__(model, resume, config, logger_path='%s/%s.log' % (config.trainer.checkpoint_dir, config.trainer.checkpoint_dir.split('/')[-1]))
        self.config = config
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_parameters = [self._prepare_data(item) for item in self.data_loader.dataset.get_parameters()]
        self.test_parameters = [self._prepare_data(item) for item in self.test_data_loader.dataset.get_parameters()]
        self.lambda_s, self.lambda_q, self.lambda_pee, self.lambda_root, self.lambda_f, self.lambda_fc = 0.1, 1, 1.2, 0.3, 0.5, 0.5
        self.stpes = 0

    def _train_epoch(self, epoch):
        def get_velocity(motions, joint_index):
            joint_motion = motions[..., [joint_index*3, joint_index*3 + 1, joint_index*3 + 2]]
            velocity = torch.sqrt(torch.sum((joint_motion[:, 2:] - joint_motion[:, :-2])**2, dim=-1))
            return velocity
        self.model.train()
        for batch_idx, datas in enumerate(self.data_loader):
            datas = [self._prepare_data(item, _from='tensor') for item in datas]
            if self.config.trainer.use_loss_D:
                poses_2d, poses_3d, bones, contacts, alphas, proj_facters, rotations = datas
            else:
                poses_2d, poses_3d, bones, contacts, alphas, proj_facters = datas
            fake_bones, fake_rotations, fake_rotations_full, fake_pose_3d, fake_c, fake_proj = self.model.forward_fk(poses_2d, self.train_parameters)
            loss_bones = torch.mean(torch.norm(fake_bones - bones, dim=-1))
            position_weights = torch.ones((1, 17)).cuda()
            position_weights[:, [0, 3, 6, 8, 11, 14]] = self.lambda_pee
            loss_positions = torch.mean(torch.norm((fake_pose_3d.view((-1, 17, 3)) - poses_3d.view((-1, 17, 3))), dim=-1)*position_weights) if self.config.trainer.use_loss_3d else 0
            loss_root = torch.mean(torch.norm(fake_proj - proj_facters, dim=-1)) if self.config.arch.translation else 0
            loss_f = torch.mean(torch.norm(fake_c - contacts, dim=-1)) if self.config.trainer.use_loss_foot else 0
            loss_fc = (torch.mean(get_velocity(fake_pose_3d, 3)[contacts[:, 1:-1, 0] == 1] ** 2) + torch.mean(get_velocity(fake_pose_3d, 6)[contacts[:, 1:-1, 0] == 1] ** 2)) if self.config.trainer.use_loss_foot else 0
            if self.config.trainer.use_loss_D:
                G_real = self.model.D(fake_rotations)
                loss_G_GAN = torch.mean(torch.norm((G_real - 1) ** 2, dim=-1))
            else:
                loss_G_GAN = 0
            
            loss_bones = loss_bones#*self.lambda_s
            loss_G = loss_positions + loss_root*self.lambda_root + loss_f*self.lambda_f + loss_fc*self.lambda_fc + loss_G_GAN*self.lambda_q
            
            self.model.optimizer_S.zero_grad()
            loss_bones.backward()
            self.model.optimizer_S.step()
            self.model.optimizer_Q.zero_grad()
            loss_G.backward()
            self.model.optimizer_Q.step()

            if self.config.trainer.use_loss_D:
                D_real = self.model.D(rotations)
                D_fake = self.model.D(fake_rotations.detach())
                loss_D = torch.mean(torch.norm((D_real - 1) ** 2)) + torch.mean(torch.sum((D_fake) ** 2, dim=-1))
                loss_D = loss_D*self.lambda_q
                self.model.optimizer_D.zero_grad()
                loss_D.backward()
                self.model.optimizer_D.step()
            else:
                loss_D = 0

            train_log = {'loss_G': loss_G, 'loss_positions': loss_positions, 'loss_bones': loss_bones, \
                                        'loss_root': loss_root, 'loss_f': loss_f, 'loss_fc': loss_fc, \
                                        'loss_G_GAN': loss_G_GAN, 'loss_D': loss_D}

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.writer.set_step(self.stpes, mode='train')
                self.writer.set_scalars(train_log)
                training_message = 'Train Epoch: {} [{}/{} ({:.0f})%]\t'.format(epoch, self.data_loader.batch_size*batch_idx, self.data_loader.n_samples, 100.0 * batch_idx / len(self.data_loader))
                for key, value in train_log.items():
                    if value > 0:
                        training_message += '{}: {:.6f}\t'.format(key, value)
                self.train_logger.info(training_message)
            self.stpes += 1
        
        val_log = self._valid_epoch(epoch)
        self.data_loader.dataset.set_sequences()
        return val_log

    def _valid_epoch(self, epoch):
        self.model.eval()
        total_val_metrics = 0
        total_val_loss = 0
        for batch_idx, datas in enumerate(self.test_data_loader):
            datas = [self._prepare_data(item, _from='tensor') for item in datas[:-1]]
            poses_2d_pixel, poses_2d, poses_3d, bones, contacts, alphas, proj_facters = datas
            _, _, _, fake_pose_3d, _, _ = self.model.forward_fk(poses_2d, self.test_parameters)
            total_val_metrics += metric.mean_points_error(fake_pose_3d, poses_3d) * torch.mean(alphas[0]).data.cpu().numpy()
            total_val_loss += torch.mean(torch.norm(fake_pose_3d.view(-1, 17, 3) - poses_3d.view(-1, 17, 3), dim=-1)).item()
        val_log = {'val_metric': total_val_metrics/len(self.test_data_loader), 'val_loss': total_val_loss/len(self.test_data_loader),}
        self.writer.set_step(epoch, mode='valid')
        self.writer.set_scalars(val_log)
        self.train_logger.info('Eveluation: mean_points_error: {:.6f} loss: {:.6f}'.format(val_log['val_metric'], val_log['val_loss']))
        return val_log