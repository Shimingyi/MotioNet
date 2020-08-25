import os
import math
import json
import datetime
import torch
import numpy as np
from utils.util import mkdir_dir
from utils.visualization import WriterTensorboardX
from utils.logger import Logger

class base_trainer:
    def __init__(self, model, resume, config, logger_path):
        self.config = config

        self.device, device_ids = self._prepare_device(config.device)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.epochs = config.trainer.epochs
        self.save_freq = config.trainer.save_freq
        self.verbosity = config.trainer.verbosity

        self.checkpoint_dir = config.trainer.checkpoint_dir
        mkdir_dir(self.checkpoint_dir)
        self.train_logger = Logger(logger_path)

        self.monitor = config.trainer.monitor
        self.monitor_mode = config.trainer.monitor_mode
        assert self.monitor_mode in ['min', 'max', 'off']
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.start_epoch = 1
        
        self.writer = WriterTensorboardX(config.trainer.checkpoint_dir, self.train_logger, config.visualization.tensorboardX)
        if resume:
            self._resume_checkpoint(resume)
    
    def _prepare_device(self, gpu_id):
        """ 
        setup GPU device if available, move model into configured device
        """ 
        n_gpu = torch.cuda.device_count()
        if n_gpu == 0:
            self.train_logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        if gpu_id > n_gpu:
            msg = "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu)
            self.train_logger.warning(msg)
            gpu_id = n_gpu
        device = torch.device('cuda:0' if gpu_id is not None else 'cpu')
        list_ids = list([gpu_id])
        return device, list_ids

    def _prepare_data(self, data, _from='numpy'):
        return torch.from_numpy(np.array(data)).float().to(self.device)if _from == 'numpy' else data.float().to(self.device) 

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({key: value})
                elif key == 'val_metrics':
                    log.update({'val_' + key: value})
                else:
                    log[key] = value

            # print logged informations to the screen
            if self.train_logger is not None:
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.train_logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.monitor_mode != 'off':
                try:
                    if  (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best) or\
                        (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
                        self.monitor_best = log[self.monitor]
                        best = True
                except KeyError:
                    if epoch == 1:
                        msg = "Warning: Can\'t recognize metric named '{}' ".format(self.monitor)\
                            + "for performance monitoring. model_best checkpoint won\'t be updated."
                        self.train_logger.warning(msg)
            if best and epoch>5:
                self._save_checkpoint('best', epoch)
            if epoch % self.save_freq == 0:
                self._save_checkpoint('epoch%s' % epoch, epoch)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, save_name, epoch):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, '%s.pth' % save_name)
        torch.save(state, filename)
        self.train_logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.train_logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint.config.arch != self.config.arch:
            self.train_logger.warning('Warning: Architecture configuration given in config file is different from that of checkpoint. ' + \
                                'This may yield an exception while state_dict is being loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])

        self.train_logger = checkpoint['logger']
        self.train_logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
