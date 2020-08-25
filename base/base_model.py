import logging
import torch
import torch.nn as nn
import numpy as np

logging.basicConfig(level = logging.INFO,format = '')

class base_model(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super(base_model, self).__init__()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.Tensor = torch.cuda.FloatTensor # if self.gpu_ids else torch.Tensor

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)