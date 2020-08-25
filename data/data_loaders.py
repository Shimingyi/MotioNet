from base.base_data_loader import BaseDataLoader

from data import h36m_dataset


NUMBER_WORK = 8


class h36m_loader(BaseDataLoader):
    def __init__(self, config, is_training=False):
        self.dataset = h36m_dataset.h36m_dataset(config, is_train=is_training)
        batch_size = config.trainer.batch_size if is_training else 1
        super(h36m_loader, self).__init__(self.dataset, batch_size=batch_size, shuffle=is_training, pin_memory=True, num_workers=NUMBER_WORK)
