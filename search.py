# -*- coding: utf-8 -*-
# @Date    : 2019-09-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from __future__ import absolute_import, division, print_function

import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

import cfg
import datasets
import models_search  # noqa
from functions import get_topk_arch_hidden, train_controller, train_shared
from utils.fid_score import check_or_download_inception, create_inception_graph
from utils.inception_score import _init_inception
from utils.utils import create_logger, RunningStats, save_checkpoint, set_log_dir

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class GrowCtrler(object):
    def __init__(self, grow_step1, grow_step2):
        self.grow_step1 = grow_step1
        self.grow_step2 = grow_step2

    def cur_stage(self, search_iter):
        """
        Return current stage.
        :param epoch: current epoch.
        :return: current stage
        """
        if search_iter < self.grow_step1:
            return 0
        elif self.grow_step1 <= search_iter < self.grow_step2:
            return 1
        else:
            return 2


def create_ctrler(args, cur_stage, weights_init):
    controller = eval("models_search." + args.controller + ".Controller")(
        args=args, cur_stage=cur_stage
    ).cuda()
    controller.apply(weights_init)
    ctrl_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, controller.parameters()),
        args.ctrl_lr,
        (args.beta1, args.beta2),
    )
    return controller, ctrl_optimizer


def create_shared_gan(args, weights_init):
    gen_net = eval("models_search." + args.gen_model + ".Generator")(args=args).cuda()
    dis_net = eval("models_search." + args.dis_model + ".Discriminator")(
        args=args
    ).cuda()
    gen_net.apply(weights_init)
    dis_net.apply(weights_init)
    gen_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gen_net.parameters()),
        args.g_lr,
        (args.beta1, args.beta2),
    )
    dis_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, dis_net.parameters()),
        args.d_lr,
        (args.beta1, args.beta2),
    )
    return gen_net, dis_net, gen_optimizer, dis_optimizer


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            if args.init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == "orth":
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == "xavier_uniform":
                nn.init.xavier_uniform(m.weight.data, 1.0)
            else:
                raise NotImplementedError(
                    "{} unknown inital type".format(args.init_type)
                )
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net, dis_net, gen_optimizer, dis_optimizer = create_shared_gan(
        args, weights_init
    )

    # set grow controller
    grow_ctrler = GrowCtrler(args.grow_step1, args.grow_step2)

    # initial
    start_search_iter = 0

    # set writer
    if args.load_path:
        print(f"=> resuming from {args.load_path}")
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path, "Model", "checkpoint.pth")
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        # set controller && its optimizer
        cur_stage = checkpoint["cur_stage"]
        controller, ctrl_optimizer = create_ctrler(args, cur_stage, weights_init)

        start_search_iter = checkpoint["search_iter"]
        gen_net.load_state_dict(checkpoint["gen_state_dict"])
        dis_net.load_state_dict(checkpoint["dis_state_dict"])
        controller.load_state_dict(checkpoint["ctrl_state_dict"])
        gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
        dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])
        ctrl_optimizer.load_state_dict(checkpoint["ctrl_optimizer"])
        prev_archs = checkpoint["prev_archs"]
        prev_hiddens = checkpoint["prev_hiddens"]

        args.path_helper = checkpoint["path_helper"]
        logger = create_logger(args.path_helper["log_path"])
        logger.info(
            f"=> loaded checkpoint {checkpoint_file} (search iteration {start_search_iter})"
        )
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir("logs", args.exp_name)
        logger = create_logger(args.path_helper["log_path"])
        prev_archs = None
        prev_hiddens = None

        # set controller && its optimizer
        cur_stage = 0
        controller, ctrl_optimizer = create_ctrler(args, cur_stage, weights_init)

    # set up data_loader
    dataset = datasets.ImageDataset(args, 2 ** (cur_stage + 3))
    train_loader = dataset.train

    logger.info(args)
    writer_dict = {
        "writer": SummaryWriter(args.path_helper["log_path"]),
        "controller_steps": start_search_iter * args.ctrl_step,
    }

    g_loss_history = RunningStats(args.dynamic_reset_window)
    d_loss_history = RunningStats(args.dynamic_reset_window)

    # train loop
    for search_iter in tqdm(
        range(int(start_search_iter), int(args.max_search_iter)), desc="search progress"
    ):
        logger.info(f"<start search iteration {search_iter}>")
        if search_iter == args.grow_step1 or search_iter == args.grow_step2:

            # save
            cur_stage = grow_ctrler.cur_stage(search_iter)
            logger.info(f"=> grow to stage {cur_stage}")
            prev_archs, prev_hiddens = get_topk_arch_hidden(
                args, controller, gen_net, prev_archs, prev_hiddens
            )

            # grow section
            del controller
            del ctrl_optimizer
            controller, ctrl_optimizer = create_ctrler(args, cur_stage, weights_init)

            dataset = datasets.ImageDataset(args, 2 ** (cur_stage + 3))
            train_loader = dataset.train

        dynamic_reset = train_shared(
            args,
            gen_net,
            dis_net,
            g_loss_history,
            d_loss_history,
            controller,
            gen_optimizer,
            dis_optimizer,
            train_loader,
            prev_hiddens=prev_hiddens,
            prev_archs=prev_archs,
        )
        train_controller(
            args,
            controller,
            ctrl_optimizer,
            gen_net,
            prev_hiddens,
            prev_archs,
            writer_dict,
        )

        if dynamic_reset:
            logger.info("re-initialize share GAN")
            del gen_net, dis_net, gen_optimizer, dis_optimizer
            gen_net, dis_net, gen_optimizer, dis_optimizer = create_shared_gan(
                args, weights_init
            )

        save_checkpoint(
            {
                "cur_stage": cur_stage,
                "search_iter": search_iter + 1,
                "gen_model": args.gen_model,
                "dis_model": args.dis_model,
                "controller": args.controller,
                "gen_state_dict": gen_net.state_dict(),
                "dis_state_dict": dis_net.state_dict(),
                "ctrl_state_dict": controller.state_dict(),
                "gen_optimizer": gen_optimizer.state_dict(),
                "dis_optimizer": dis_optimizer.state_dict(),
                "ctrl_optimizer": ctrl_optimizer.state_dict(),
                "prev_archs": prev_archs,
                "prev_hiddens": prev_hiddens,
                "path_helper": args.path_helper,
            },
            False,
            args.path_helper["ckpt_path"],
        )

    final_archs, _ = get_topk_arch_hidden(
        args, controller, gen_net, prev_archs, prev_hiddens
    )
    logger.info(f"discovered archs: {final_archs}")


if __name__ == "__main__":
    main()
