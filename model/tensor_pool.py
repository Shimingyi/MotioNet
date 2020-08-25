import random
import torch
from torch.autograd import Variable
class TensorPool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, tensors):
        if self.pool_size == 0:
            return tensors
        return_tensors = []
        for tensor in tensors.data:
            tensor = torch.unsqueeze(tensor, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(tensor)
                return_tensors.append(tensor)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = tensor
                    return_tensors.append(tmp)
                else:
                    return_tensors.append(tensor)
        return_images = Variable(torch.cat(return_tensors, 0))
        return return_images
