import argparse
import os
import random

import numpy
import torch
from torch.backends import cudnn

from trainer import Trainer
from utils import TBLog, get_logger, over_write_args


def main(opt):
    # random seed
    assert opt.seed is not None
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    numpy.random.seed(opt.seed)
    cudnn.deterministic = True

    save_path = os.path.join(opt.save_dir, opt.save_name)
    opt.save_path = save_path
    if os.path.exists(save_path) and not opt.overwrite:
        raise Exception('already existing model: {}'.format(save_path))

    # set logger
    tb_logger = TBLog(save_path, 'tensorboard', True)
    logger_level = "INFO"
    logger = get_logger(opt.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {opt.gpu} for training")
    logger.info(opt)
    trainer = Trainer(args=opt, tb_logger=tb_logger, logger=logger)
    trainer()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TR-CLIP-Captioning!')
    parser.add_argument('--config', type=str, default='./configs/tasviret/tasviret_exp1_base16.yaml')
    args = parser.parse_args()
    over_write_args(args, args.config)
    main(args)
