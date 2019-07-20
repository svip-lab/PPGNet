# System libs
import os
import time

# Numerical libs
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

# Our libs
from data.sist_line import SISTLine
import data.transforms as tf
from models.lsd import LSDModule
from utils import AverageMeter, graph2line, draw_lines, draw_jucntions

# tensorboard
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

import fire


def weight_fn(dist_map, max_dist, mid=0.1, scale=10):
    with torch.no_grad():
        dist_map = dist_map / max_dist
        weight = (torch.exp(scale * (dist_map - mid)) - torch.exp(scale * (-dist_map + mid))) / \
                   (torch.exp(scale * (dist_map - mid)) + torch.exp(scale * (-dist_map + mid))) / 2 + 0.5
        return weight


class LSDTrainer(object):
    def __init__(
            self,
            # exp params
            exp_name="u50_block",
            # arch params
            backbone="resnet50",
            backbone_kwargs={},
            dim_embedding=256,
            feature_spatial_scale=0.25,
            max_junctions=512,
            junction_pooling_threshold=0.2,
            junc_pooling_size=15,
            attention_sigma=1.,
            junction_heatmap_criterion="binary_cross_entropy",
            block_inference_size=64,
            adjacency_matrix_criterion="binary_cross_entropy",
            # data params
            data_root=r"/home/ziheng/indoorDist_new2",
            img_size=416,
            junc_sigma=3.,
            batch_size=2,
            # train params
            gpus=[0,],
            num_workers=5,
            resume_epoch="latest",
            is_train_junc=True,
            is_train_adj=True,
            # vis params
            vis_junc_th=0.3,
            vis_line_th=0.3
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(c) for c in gpus)

        self.is_cuda = bool(gpus)

        self.model = LSDModule(
            backbone=backbone,
            dim_embedding=dim_embedding,
            backbone_kwargs=backbone_kwargs,
            junction_pooling_threshold=junction_pooling_threshold,
            max_junctions=max_junctions,
            feature_spatial_scale=feature_spatial_scale,
            junction_heatmap_criterion=junction_heatmap_criterion,
            junction_pooling_size=junc_pooling_size,
            attention_sigma=attention_sigma,
            block_inference_size=block_inference_size,
            adjacency_matrix_criterion=adjacency_matrix_criterion,
            weight_fn=weight_fn,
            is_train_adj=is_train_adj,
            is_train_junc=is_train_junc
        )

        self.exp_name = exp_name
        os.makedirs(os.path.join("log", exp_name), exist_ok=True)
        os.makedirs(os.path.join("ckpt", exp_name), exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join("log", exp_name))

        # checkpoints
        self.states = dict(
            last_epoch=-1,
            elapsed_time=0,
            state_dict=None
        )

        if resume_epoch and os.path.isfile(os.path.join("ckpt", exp_name, f"train_states_{resume_epoch}.pth")):
            states = torch.load(
                os.path.join("ckpt", exp_name, f"train_states_{resume_epoch}.pth"))
            print(f"resume traning from epoch {states['last_epoch']}")
            self.model.load_state_dict(states["state_dict"])
            self.states.update(states)

        self.train_data = SISTLine(
            data_root=data_root,
            transforms=tf.Compose(
                tf.Resize((img_size, img_size)),
                tf.RandomHorizontalFlip(),
                tf.RandomColorAug()
            ),
            phase="train",
            sigma_junction=junc_sigma,
            max_junctions=max_junctions)
                  
        assert len(self.train_data) > 0, "Wow, there is nothing in your data folder. Please check the --data-root parameter in your train.sh."

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        self.eval_data = SISTLine(
            data_root=data_root,
            transforms=tf.Compose(
                tf.Resize((img_size, img_size)),
            ),
            phase="val",
            sigma_junction=junc_sigma,
            max_junctions=max_junctions)

        self.eval_loader = DataLoader(
            self.eval_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        self.vis_junc_th = vis_junc_th
        self.vis_line_th = vis_line_th
        self.block_size = block_inference_size
        self.max_junctions = max_junctions
        self.is_train_junc = is_train_junc
        self.is_train_adj = is_train_adj

    @staticmethod
    def _group_weight(module, lr):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.batchnorm._BatchNorm) or isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)

        assert len(list(
            module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [
            dict(params=group_decay, lr=lr),
            dict(params=group_no_decay, lr=lr, weight_decay=.0)
        ]
        return groups

    def end(self):
        self.writer.close()
        return "command queue finished."

    def _train_epoch(self):
        net_time = AverageMeter()
        data_time = AverageMeter()
        vis_time = AverageMeter()

        epoch = self.states["last_epoch"]
        data_loader = self.train_loader
        if self.is_cuda:
            self.model = self.model.cuda()
        params = self._group_weight(self.model.backbone, self.lr)
        if self.is_train_junc:
            params += self._group_weight(self.model.junc_infer, self.lr)
        if self.is_train_adj:
            params += self._group_weight(self.model.adj_infer, self.lr)
            params += self._group_weight(self.model.adj_embed, self.lr)
        if self.solver == "Adadelta":
            solver = optim.__dict__[self.solver](params, weight_decay=self.weight_decay)
        else:
            solver = optim.__dict__[self.solver](params, weight_decay=self.weight_decay, momentum=self.momentum)

        # main loop
        torch.set_grad_enabled(True)
        tic = time.time()
        print(f"start training epoch: {epoch}", flush=True)

        if self.is_cuda:
            model = nn.DataParallel(self.model).train()
        else:
            model = self.model.train()

        for i, batch in enumerate(data_loader):
            if self.is_cuda:
                img = batch["image"].cuda()
                heatmap_gt = batch["heatmap"].cuda()
                adj_mtx_gt = batch["adj_mtx"].cuda()
                junctions_gt = batch["junctions"].cuda()
            else:
                img = batch["image"]
                heatmap_gt = batch["heatmap"]
                adj_mtx_gt = batch["adj_mtx"]
                junctions_gt = batch["junctions"]

            # measure elapsed time
            data_time.update(time.time() - tic)
            tic = time.time()

            junc_pred, heatmap_pred, adj_mtx_pred, loss_hm, loss_adj = model(
                img, heatmap_gt, adj_mtx_gt, self.lambda_heatmap, self.lambda_adj, junctions_gt
            )

            model.zero_grad()
            loss_adj = loss_adj.mean()
            loss_hm = loss_hm.mean()
            loss = (loss_hm if self.is_train_junc else 0) + (loss_adj if self.is_train_adj else 0)
            loss.backward()
            solver.step()

            # measure elapsed time
            net_time.update(time.time() - tic)
            tic = time.time()

            # visualize result
            if i % self.vis_line_interval == 0:
                img = img.cpu().numpy()
                heatmap_pred = heatmap_pred.detach().cpu()
                adj_mtx_pred = adj_mtx_pred.detach().cpu().numpy()
                junctions_gt = junctions_gt.cpu().numpy()
                adj_mtx_gt = adj_mtx_gt.cpu().numpy()
                self._vis_train(epoch, i, len(data_loader), img, heatmap_pred, adj_mtx_pred, junctions_gt, adj_mtx_gt)

            vis_heatmap_gt = vutils.make_grid(
                heatmap_gt.view(heatmap_gt.size(0), 1, heatmap_gt.size(1), heatmap_gt.size(2)))
            vis_heatmap_pred = vutils.make_grid(
                heatmap_pred.view(heatmap_gt.size(0), 1, heatmap_gt.size(1), heatmap_gt.size(2)))

            self.writer.add_scalar(self.exp_name + "/" + "train/loss_total",
                                   loss.item(),
                                   epoch * len(data_loader) + i)
            self.writer.add_scalar(self.exp_name + "/" + "train/loss_heatmap",
                                   loss_hm.item() / self.lambda_heatmap if self.lambda_heatmap else 0,
                                   epoch * len(data_loader) + i)
            self.writer.add_scalar(self.exp_name + "/" + "train/loss_adj_mtx",
                                   loss_adj.item() / self.lambda_adj if self.lambda_adj else 0,
                                   epoch * len(data_loader) + i)
            self.writer.add_image(self.exp_name + "/" + "train/heatmap_gt",
                                  vis_heatmap_gt,
                                  epoch * len(data_loader) + i)
            self.writer.add_image(self.exp_name + "/" + "train/heatmap_pred",
                                  vis_heatmap_pred,
                                  epoch * len(data_loader) + i)

            vis_time.update(time.time() - tic)
            info = f"epoch: [{epoch}][{i}/{len(data_loader)}], lr: {self.lr}, " \
                   f"time_total: {net_time.average() + data_time.average() + vis_time.average():.2f}, " \
                   f"time_data: {data_time.average():.2f}, time_net: {net_time.average():.2f}, " \
                   f"time_vis: {vis_time.average():.2f}, " \
                   f"loss: {loss.item():.4f}, " \
                   f"loss_heatmap: {loss_hm.item() / self.lambda_heatmap if self.lambda_heatmap else 0:.4f}, " \
                   f"loss_adj_mtx: {loss_adj.item() / self.lambda_adj if self.lambda_adj else 0:.4f}"
            self.writer.add_text(self.exp_name + "/" + "train/info", info,
                                 epoch * len(data_loader) + i)
            print(info, flush=True)
            # measure elapsed time
            tic = time.time()

    def _vis_train(self, epoch, i, len_loader, img, heatmap, adj_mtx, junctions_gt, adj_mtx_gt):
        junctions_gt = np.int32(junctions_gt)
        lines_gt, scores_gt = graph2line(junctions_gt, adj_mtx_gt)
        vis_line_gt = vutils.make_grid(
            draw_lines(img, lines_gt, scores_gt))
        lines_pred, score_pred = graph2line(junctions_gt, adj_mtx, threshold=self.vis_line_th)
        vis_line_pred = vutils.make_grid(
            draw_lines(img, lines_pred, score_pred))
        junc_score = []
        line_score = []
        for m, juncs in zip(heatmap, junctions_gt):
            juncs = juncs[juncs.sum(axis=1) > 0]
            junc_score += m[juncs[:, 1], juncs[:, 0]].tolist()
        for s in score_pred:
            line_score += s.tolist()

        self.writer.add_image(self.exp_name + "/" + "train/lines_gt",
                              vis_line_gt,
                              epoch * len_loader + i)
        self.writer.add_image(self.exp_name + "/" + "train/lines_pred",
                              vis_line_pred,
                              epoch * len_loader + i)
        self.writer.add_scalar(
            self.exp_name + "/" + "train/mean_junc_score",
            np.mean(junc_score),
            epoch * len_loader + i)
        self.writer.add_scalar(
            self.exp_name + "/" + "train/mean_line_score",
            np.mean(line_score),
            epoch * len_loader + i)

    def _checkpoint(self):
        print('Saving checkpoints...')

        train_states = self.states

        train_states["state_dict"] = self.model.cpu().state_dict()

        torch.save(
            train_states,
            os.path.join("ckpt", self.exp_name,
                         "train_states_latest.pth"))
        torch.save(
            train_states,
            os.path.join("ckpt", self.exp_name,
                         f"train_states_{self.states['last_epoch']}.pth"))

        state = torch.load(os.path.join("ckpt", self.exp_name, "train_states_latest.pth"))
        self.model.load_state_dict(state["state_dict"])

    def train(
            self,
            end_epoch=20,
            solver="SGD",
            lr=1.,
            weight_decay=5e-4,
            momentum=0.9,
            lambda_heatmap=1.,
            lambda_adj=1.,
            vis_line_interval=20,
    ):
        self.vis_line_interval = vis_line_interval
        self.end_epoch = end_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lambda_heatmap = lambda_heatmap
        self.lambda_adj = lambda_adj
        self.solver = solver

        start_epoch = self.states["last_epoch"] + 1

        for epoch in range(start_epoch, end_epoch):
            self.states["last_epoch"] = epoch
            self._train_epoch()
            self._checkpoint()

        return self

    def _vis_eval(self, epoch, i, len_loader, img, heatmap, adj_mtx, junctions_pred, junctions_gt, adj_mtx_gt):
        junctions_gt = np.int32(junctions_gt)
        lines_gt, scores_gt = graph2line(junctions_gt, adj_mtx_gt, threshold=self.vis_junc_th)
        vis_line_gt = vutils.make_grid(
            draw_lines(img, lines_gt, scores_gt))
        img_with_junc = draw_jucntions(img, junctions_pred)
        img_with_junc = torch.stack(img_with_junc, dim=0).numpy()[:, ::-1, :, :]
        lines_pred, score_pred = graph2line(junctions_pred, adj_mtx)
        vis_line_pred = vutils.make_grid(
            draw_lines(img_with_junc, lines_pred, score_pred))
        junc_score = []
        line_score = []
        for m, juncs in zip(heatmap, junctions_gt):
            juncs = juncs[juncs.sum(axis=1) > 0]
            junc_score += m[juncs[:, 1], juncs[:, 0]].tolist()
        for s in score_pred:
            line_score += s.tolist()

        junc_pooling = vutils.make_grid(draw_jucntions(heatmap, junctions_pred))

        self.writer.add_image(self.exp_name + "/" + "eval/junction_pooling",
                              junc_pooling,
                              epoch * len_loader + i)

        self.writer.add_image(self.exp_name + "/" + "eval/lines_gt",
                              vis_line_gt,
                              epoch * len_loader + i)
        self.writer.add_image(self.exp_name + "/" + "eval/lines_pred",
                              vis_line_pred,
                              epoch * len_loader + i)
        self.writer.add_scalar(
            self.exp_name + "/" + "eval/mean_junc_score",
            np.mean(junc_score),
            epoch * len_loader + i)
        self.writer.add_scalar(
            self.exp_name + "/" + "eval/mean_line_score",
            np.mean(line_score),
            epoch * len_loader + i)

    def eval(self,
             lambda_heatmap=1.,
             lambda_adj=1.,
             off_line=False,
             epoch=None
             ):

        if not off_line:
            if not (self.states["last_epoch"] == epoch - 1):
                return self
        else:
            self.lambda_heatmap = lambda_heatmap
            self.lambda_adj = lambda_adj

        net_time = AverageMeter()
        data_time = AverageMeter()
        vis_time = AverageMeter()
        ave_loss = AverageMeter()
        ave_loss_heatmap = AverageMeter()
        ave_loss_adj_mtx = AverageMeter()

        epoch = self.states["last_epoch"]
        data_loader = self.eval_loader

        # main loop
        torch.set_grad_enabled(False)
        tic = time.time()
        print(f"start evaluating epoch: {epoch}", flush=True)

        if self.is_cuda:
            model = nn.DataParallel(self.model.cuda()).train()
        else:
            model = self.model.train()

        for i, batch in enumerate(data_loader):
            if self.is_cuda:
                img = batch["image"].cuda()
                heatmap_gt = batch["heatmap"].cuda()
                adj_mtx_gt = batch["adj_mtx"].cuda()
                junctions_gt = batch["junctions"].cuda()
            else:
                img = batch["image"]
                heatmap_gt = batch["heatmap"]
                adj_mtx_gt = batch["adj_mtx"]
                junctions_gt = batch["junctions"]

            # measure elapsed time
            data_time.update(time.time() - tic)
            tic = time.time()

            junc_pred, heatmap_pred, adj_mtx_pred, loss_hm, loss_adj = model(
                img, heatmap_gt, adj_mtx_gt, self.lambda_heatmap, self.lambda_adj, junctions_gt
            )

            loss_adj = loss_adj.mean()
            loss_hm = loss_hm.mean()
            loss = loss_adj + loss_hm
            ave_loss_adj_mtx.update(loss_adj.item() / self.lambda_adj if self.lambda_adj else 0)
            ave_loss_heatmap.update(loss_hm.item() / self.lambda_heatmap if self.lambda_heatmap else 0)
            ave_loss.update(loss.item())

            # measure elapsed time
            net_time.update(time.time() - tic)
            tic = time.time()

            # visualize eval
            img = img.cpu().numpy()
            heatmap = heatmap_pred.detach().cpu().numpy()
            junctions_pred = junc_pred.detach().cpu().numpy()
            adj_mtx = adj_mtx_pred.detach().cpu().numpy()
            junctions_gt = junctions_gt.cpu().numpy()
            adj_mtx_gt = adj_mtx_gt.cpu().numpy()
            self._vis_eval(epoch, i, len(data_loader), img, heatmap, adj_mtx, junctions_pred, junctions_gt, adj_mtx_gt)

            vis_heatmap_gt = vutils.make_grid(
                heatmap_gt.view(heatmap_gt.size(0), 1, heatmap_gt.size(1), heatmap_gt.size(2)))
            vis_heatmap_pred = vutils.make_grid(
                heatmap.view(heatmap_gt.size(0), 1, heatmap_gt.size(1), heatmap_gt.size(2)))

            self.writer.add_scalar(self.exp_name + "/" + "eval/loss_total",
                                   loss.item(),
                                   epoch * len(data_loader) + i)
            self.writer.add_scalar(self.exp_name + "/" + "eval/loss_heatmap",
                                   loss_hm.item() / self.lambda_heatmap if self.lambda_heatmap else 0,
                                   epoch * len(data_loader) + i)
            self.writer.add_scalar(self.exp_name + "/" + "eval/loss_adj_mtx",
                                   loss_adj.item() / self.lambda_adj if self.lambda_adj else 0,
                                   epoch * len(data_loader) + i)
            self.writer.add_image(self.exp_name + "/" + "eval/heatmap_gt",
                                  vis_heatmap_gt,
                                  epoch * len(data_loader) + i)
            self.writer.add_image(self.exp_name + "/" + "eval/heatmap_pred",
                                  vis_heatmap_pred,
                                  epoch * len(data_loader) + i)

            vis_time.update(time.time() - tic)
            info = f"epoch: [{epoch}][{i}/{len(data_loader)}], " \
                   f"time_total: {net_time.average() + data_time.average() + vis_time.average():.2f}, " \
                   f"time_data: {data_time.average():.2f}, time_net: {net_time.average():.2f}, " \
                   f"time_vis: {vis_time.average():.2f}, " \
                   f"loss: {loss.item():.4f}, " \
                   f"loss_heatmap: {loss_hm.item() / self.lambda_heatmap if self.lambda_heatmap else 0:.4f}, " \
                   f"loss_adj_mtx: {loss_adj.item() / self.lambda_adj if self.lambda_adj else 0:.4f}"
            if i == len(data_loader) - 1:
                info += f"\n*[{epoch}] " \
                        f"ave_loss: {ave_loss.average():.4f}, " \
                        f"ave_loss_heatmap: {ave_loss_heatmap.average():.4f}, " \
                        f"ave_loss_adj_mtx: {ave_loss_adj_mtx.average():.4f}"

            self.writer.add_text(self.exp_name + "/" + "eval/info", info,
                                 epoch * len(data_loader) + i)
            print(info, flush=True)
            # measure elapsed time
            tic = time.time()

        return self


if __name__ == "__main__":
    fire.Fire(LSDTrainer)
    # trainer = LSDTrainer().train(lr=1.)
