# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import torch
from tensorboardX import SummaryWriter

import cfg
import models  # noqa
from functions import validate
from utils.fid_score import check_or_download_inception, create_inception_graph
from utils.inception_score import _init_inception
from utils.utils import create_logger, set_log_dir

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    assert args.exp_name
    assert args.load_path.endswith(".pth")
    assert os.path.exists(args.load_path)
    args.path_helper = set_log_dir("logs_eval", args.exp_name)
    logger = create_logger(args.path_helper["log_path"], phase="test")

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # import network
    gen_net = eval("models." + args.gen_model + ".Generator")(args=args).cuda()

    # fid stat
    if args.dataset.lower() == "cifar10":
        fid_stat = "fid_stat/fid_stats_cifar10_train.npz"
    elif args.dataset.lower() == "stl10":
        fid_stat = "fid_stat/stl10_train_unlabeled_fid_stats_48.npz"
    else:
        raise NotImplementedError(f"no fid stat for {args.dataset.lower()}")
    assert os.path.exists(fid_stat)

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))

    # set writer
    logger.info(f"=> resuming from {args.load_path}")
    checkpoint_file = args.load_path
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)

    if "avg_gen_state_dict" in checkpoint:
        gen_net.load_state_dict(checkpoint["avg_gen_state_dict"])
        epoch = checkpoint["epoch"]
        logger.info(f"=> loaded checkpoint {checkpoint_file} (epoch {epoch})")
    else:
        gen_net.load_state_dict(checkpoint)
        logger.info(f"=> loaded checkpoint {checkpoint_file}")

    logger.info(args)
    writer_dict = {
        "writer": SummaryWriter(args.path_helper["log_path"]),
        "valid_global_steps": 0,
    }
    inception_score, fid_score = validate(
        args, fixed_z, fid_stat, gen_net, writer_dict, clean_dir=False
    )
    logger.info(f"Inception score: {inception_score}, FID score: {fid_score}.")


if __name__ == "__main__":
    main()
