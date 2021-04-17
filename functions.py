# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import logging
import operator
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from imageio import imsave
from torchvision.utils import make_grid
from tqdm import tqdm

from utils.fid_score import calculate_fid_given_paths
from utils.inception_score import get_inception_score

logger = logging.getLogger(__name__)


def train_shared(
    args,
    gen_net: nn.Module,
    dis_net: nn.Module,
    g_loss_history,
    d_loss_history,
    controller,
    gen_optimizer,
    dis_optimizer,
    train_loader,
    prev_hiddens=None,
    prev_archs=None,
):
    dynamic_reset = False
    logger.info("=> train shared GAN...")
    step = 0
    gen_step = 0

    # train mode
    gen_net.train()
    dis_net.train()

    # eval mode
    controller.eval()
    for epoch in range(args.shared_epoch):
        for iter_idx, (imgs, _) in enumerate(train_loader):

            # sample an arch
            arch = controller.sample(
                1, prev_hiddens=prev_hiddens, prev_archs=prev_archs
            )[0][0]
            gen_net.set_arch(arch, controller.cur_stage)
            dis_net.cur_stage = controller.cur_stage
            # Adversarial ground truths
            real_imgs = imgs.type(torch.cuda.FloatTensor)

            # Sample noise as generator input
            z = torch.cuda.FloatTensor(
                np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))
            )

            # ---------------------
            #  Train Discriminator
            # ---------------------
            dis_optimizer.zero_grad()

            real_validity = dis_net(real_imgs)
            fake_imgs = gen_net(z).detach()
            assert fake_imgs.size() == real_imgs.size(), print(
                f"fake image size is {fake_imgs.size()}, "
                f"while real image size is {real_imgs.size()}"
            )

            fake_validity = dis_net(fake_imgs)

            # cal loss
            d_loss = torch.mean(
                nn.ReLU(inplace=True)(1.0 - real_validity)
            ) + torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
            d_loss.backward()
            dis_optimizer.step()

            # add to window
            d_loss_history.push(d_loss.item())

            # -----------------
            #  Train Generator
            # -----------------
            if step % args.n_critic == 0:
                gen_optimizer.zero_grad()

                gen_z = torch.cuda.FloatTensor(
                    np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim))
                )
                gen_imgs = gen_net(gen_z)
                fake_validity = dis_net(gen_imgs)

                # cal loss
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                gen_optimizer.step()

                # add to window
                g_loss_history.push(g_loss.item())
                gen_step += 1

            # verbose
            if gen_step and iter_idx % args.print_freq == 0:
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (
                        epoch,
                        args.shared_epoch,
                        iter_idx % len(train_loader),
                        len(train_loader),
                        d_loss.item(),
                        g_loss.item(),
                    )
                )

            # check window
            if g_loss_history.is_full():
                if (
                    g_loss_history.get_var() < args.dynamic_reset_threshold
                    or d_loss_history.get_var() < args.dynamic_reset_threshold
                ):
                    dynamic_reset = True
                    logger.info("=> dynamic resetting triggered")
                    g_loss_history.clear()
                    d_loss_history.clear()
                    return dynamic_reset

            step += 1

    return dynamic_reset


def train(
    args,
    gen_net: nn.Module,
    dis_net: nn.Module,
    gen_optimizer,
    dis_optimizer,
    gen_avg_param,
    train_loader,
    epoch,
    writer_dict,
    schedulers=None,
):
    writer = writer_dict["writer"]
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict["train_global_steps"]

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(
            np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))
        )

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + torch.mean(
            nn.ReLU(inplace=True)(1 + fake_validity)
        )
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar("d_loss", d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(
                np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim))
            )
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)

            # cal loss
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar("LR/g_lr", g_lr, global_steps)
                writer.add_scalar("LR/d_lr", d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar("g_loss", g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    args.max_epoch,
                    iter_idx % len(train_loader),
                    len(train_loader),
                    d_loss.item(),
                    g_loss.item(),
                )
            )

        writer_dict["train_global_steps"] = global_steps + 1


def train_controller(
    args, controller, ctrl_optimizer, gen_net, prev_hiddens, prev_archs, writer_dict
):
    logger.info("=> train controller...")
    writer = writer_dict["writer"]
    baseline = None

    # train mode
    controller.train()

    # eval mode
    gen_net.eval()

    cur_stage = controller.cur_stage
    for step in range(args.ctrl_step):
        controller_step = writer_dict["controller_steps"]
        archs, selected_log_probs, entropies = controller.sample(
            args.ctrl_sample_batch, prev_hiddens=prev_hiddens, prev_archs=prev_archs
        )
        cur_batch_rewards = []
        for arch in archs:
            logger.info(f"arch: {arch}")
            gen_net.set_arch(arch, cur_stage)
            is_score = get_is(args, gen_net, args.rl_num_eval_img)
            logger.info(f"get Inception score of {is_score}")
            cur_batch_rewards.append(is_score)
        cur_batch_rewards = torch.tensor(cur_batch_rewards, requires_grad=False).cuda()
        cur_batch_rewards = (
            cur_batch_rewards.unsqueeze(-1) + args.entropy_coeff * entropies
        )  # bs * 1
        if baseline is None:
            baseline = cur_batch_rewards
        else:
            baseline = (
                args.baseline_decay * baseline.detach()
                + (1 - args.baseline_decay) * cur_batch_rewards
            )
        adv = cur_batch_rewards - baseline

        # policy loss
        loss = -selected_log_probs * adv
        loss = loss.sum()

        # update controller
        ctrl_optimizer.zero_grad()
        loss.backward()
        ctrl_optimizer.step()

        # write
        mean_reward = cur_batch_rewards.mean().item()
        mean_adv = adv.mean().item()
        mean_entropy = entropies.mean().item()
        writer.add_scalar("controller/loss", loss.item(), controller_step)
        writer.add_scalar("controller/reward", mean_reward, controller_step)
        writer.add_scalar("controller/entropy", mean_entropy, controller_step)
        writer.add_scalar("controller/adv", mean_adv, controller_step)

        writer_dict["controller_steps"] = controller_step + 1


def get_is(args, gen_net: nn.Module, num_img):
    """
    Get inception score.
    :param args:
    :param gen_net:
    :param num_img:
    :return: Inception score
    """

    # eval mode
    gen_net = gen_net.eval()

    eval_iter = num_img // args.eval_batch_size
    img_list = list()
    for _ in range(eval_iter):
        z = torch.cuda.FloatTensor(
            np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim))
        )

        # Generate a batch of images
        gen_imgs = (
            gen_net(z)
            .mul_(127.5)
            .add_(127.5)
            .clamp_(0.0, 255.0)
            .permute(0, 2, 3, 1)
            .to("cpu", torch.uint8)
            .numpy()
        )
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info("calculate Inception score...")
    mean, std = get_inception_score(img_list)

    return mean


def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict, clean_dir=True):
    writer = writer_dict["writer"]
    global_steps = writer_dict["valid_global_steps"]

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper["sample_path"], "fid_buffer")
    os.makedirs(fid_buffer_dir, exist_ok=True)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc="sample images"):
        z = torch.cuda.FloatTensor(
            np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim))
        )

        # Generate a batch of images
        gen_imgs = (
            gen_net(z)
            .mul_(127.5)
            .add_(127.5)
            .clamp_(0.0, 255.0)
            .permute(0, 2, 3, 1)
            .to("cpu", torch.uint8)
            .numpy()
        )
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f"iter{iter_idx}_b{img_idx}.png")
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info("=> calculate inception score")
    mean, std = get_inception_score(img_list)
    print(f"Inception score: {mean}")

    # get fid score
    logger.info("=> calculate fid score")
    fid_score = calculate_fid_given_paths(
        [fid_buffer_dir, fid_stat], inception_path=None
    )
    print(f"FID score: {fid_score}")

    if clean_dir:
        os.system("rm -r {}".format(fid_buffer_dir))
    else:
        logger.info(f"=> sampled images are saved to {fid_buffer_dir}")

    writer.add_image("sampled_images", img_grid, global_steps)
    writer.add_scalar("Inception_score/mean", mean, global_steps)
    writer.add_scalar("Inception_score/std", std, global_steps)
    writer.add_scalar("FID_score", fid_score, global_steps)

    writer_dict["valid_global_steps"] = global_steps + 1

    return mean, fid_score


def get_topk_arch_hidden(args, controller, gen_net, prev_archs, prev_hiddens):
    """
    ~
    :param args:
    :param controller:
    :param gen_net:
    :param prev_archs: previous architecture
    :param prev_hiddens: previous hidden vector
    :return: a list of topk archs and hiddens.
    """
    logger.info(
        f"=> get top{args.topk} archs out of {args.num_candidate} candidate archs..."
    )
    assert args.num_candidate >= args.topk
    controller.eval()
    cur_stage = controller.cur_stage
    archs, _, _, hiddens = controller.sample(
        args.num_candidate,
        with_hidden=True,
        prev_archs=prev_archs,
        prev_hiddens=prev_hiddens,
    )
    hxs, cxs = hiddens
    arch_idx_perf_table = {}
    for arch_idx in range(len(archs)):
        logger.info(f"arch: {archs[arch_idx]}")
        gen_net.set_arch(archs[arch_idx], cur_stage)
        is_score = get_is(args, gen_net, args.rl_num_eval_img)
        logger.info(f"get Inception score of {is_score}")
        arch_idx_perf_table[arch_idx] = is_score
    topk_arch_idx_perf = sorted(
        arch_idx_perf_table.items(), key=operator.itemgetter(1)
    )[::-1][: args.topk]
    topk_archs = []
    topk_hxs = []
    topk_cxs = []
    logger.info(f"top{args.topk} archs:")
    for arch_idx_perf in topk_arch_idx_perf:
        logger.info(arch_idx_perf)
        arch_idx = arch_idx_perf[0]
        topk_archs.append(archs[arch_idx])
        topk_hxs.append(hxs[arch_idx].detach().requires_grad_(False))
        topk_cxs.append(cxs[arch_idx].detach().requires_grad_(False))

    return topk_archs, (topk_hxs, topk_cxs)


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
