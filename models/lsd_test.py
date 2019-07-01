import torch as th
import torch.nn as nn
import models.graph
import models.backbone
import models.common
import numpy as np

from .lsd import LSDModule


class LSDDataLayer(nn.Module):
    def __init__(self, mean=None, std=None):
        super(LSDDataLayer, self).__init__()
        self.std = [1., 1., 1.] if std is None else std
        self.mean = [102.9801, 115.9465, 122.7717] if mean is None else mean

    def forward(self, img):
        assert img.size(1) == 3
        for ch in range(3):
            img[:, ch, :, :] = (img[:, ch, :, :] - self.mean[ch]) / self.std[ch]

        return img


# noinspection PyTypeChecker
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=0.25, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target, weight=None):
        if weight is not None:
            assert weight.size() == input.size(), f"weight size: {weight.size()}, input size: {input.size()}"
            assert (weight >= 0).all() and (weight <= 1).all(), f"weight max: {weight.max()}, min: {weight.min()}"
        input = input.clamp(1.e-6, 1. - 1.e-6)
        if weight is None:
            loss = th.sum(
                - self.alpha * target * ((1 - input) ** self.gamma) * th.log(input)
                - (1 - self.alpha) * (1 - target) * (input ** self.gamma) * th.log(1 - input))
        else:
            loss = th.sum(
                (- self.alpha * target * ((1 - input) ** self.gamma) * th.log(input)
                 - (1 - self.alpha) * (1 - target) * (input ** self.gamma) * th.log(1 - input)) * weight
            )
        if self.size_average:
            loss /= input.nelement()
        return loss


class BlockAdjacencyMatrixInference(nn.Module):
    def __init__(self,
                 line_pool_module, adj_infer_module,
                 current_batch_id, junc_st_st, junc_st_len, junc_ed_st, junc_ed_len, junc_pred
                 ):
        super(BlockAdjacencyMatrixInference, self).__init__()
        self.line_pool = line_pool_module
        self.adj_infer = adj_infer_module
        self.b = current_batch_id
        self.st_st = junc_st_st
        self.st_len = junc_st_len
        self.ed_st = junc_ed_st
        self.ed_len = junc_ed_len
        self.juncs = junc_pred

    def forward(self, feat):
        junc_st = self.juncs.narrow(0, self.st_st, self.st_len)
        junc_ed = self.juncs.narrow(0, self.ed_st, self.ed_len)
        assert (junc_st[:, 0] == self.b).all() and (junc_ed[:, 0] == self.b).all(), f"{self.b}\n{junc_st[:, 0]}\n{junc_ed[:, 0]}"
        line_feat = self.line_pool(feat, junc_st, junc_ed)
        block_adj_matrix = self.adj_infer(line_feat)

        return block_adj_matrix


class BlockAdjacencyMatrixInferenceCriterion(nn.Module):
    def __init__(self, adj_matrix_crit, adj_matrix_gt, adj_matrix_loss_lambda,
                 current_batch_id, mtx_st_st, mtx_st_len, mtx_ed_st, mtx_ed_len,
                 junc_padded, img_size, line_seg_length_weight_fn
                 ):
        super(BlockAdjacencyMatrixInferenceCriterion, self).__init__()
        self.adj_crit = adj_matrix_crit
        self.adj_gt = adj_matrix_gt
        self.loss_lambda = adj_matrix_loss_lambda
        self.b = current_batch_id
        self.st_st = mtx_st_st
        self.st_len = mtx_st_len
        self.ed_st = mtx_ed_st
        self.ed_len = mtx_ed_len
        self.junc = junc_padded
        self.img_size = img_size
        self.weight = line_seg_length_weight_fn

    def forward(self, block_adj_matrix):
        block_adj_matrix_gt = self.adj_gt[self.b, self.st_st:self.st_st+self.st_len, self.ed_st:self.ed_st+self.ed_len]
        if self.junc is not None:
            junc_st = self.junc[self.b, self.st_st:self.st_st + self.st_len].view(self.st_len, 1, 2).expand(self.st_len,
                                                                                                            self.ed_len,
                                                                                                            2)
            junc_ed = self.junc[self.b, self.ed_st:self.ed_st + self.ed_len].view(1, self.ed_len, 2).expand(self.st_len,
                                                                                                            self.ed_len,
                                                                                                            2)
            line_len = (junc_ed - junc_st).norm(dim=2)
            return self.loss_lambda * self.adj_crit(block_adj_matrix, block_adj_matrix_gt, weight=self.weight(line_len, self.img_size * 1.4143))
        else:
            return self.loss_lambda * self.adj_crit(block_adj_matrix, block_adj_matrix_gt)


class LSDTestModule(LSDModule):
    def __init__(
            self,
            # backbone parameters
            backbone="unet",
            dim_embedding=256,
            backbone_kwargs={},
            # junction inference parameters
            junction_pooling_threshold=0.2,
            max_junctions=512,
            feature_spatial_scale=0.25,
            # junction pooling parameters
            junction_pooling_size=15.,
    ):
        super(LSDTestModule, self).__init__(
            backbone=backbone,
            dim_embedding=dim_embedding,
            backbone_kwargs=backbone_kwargs,
            # junction inference parameters
            junction_pooling_threshold=junction_pooling_threshold,
            max_junctions=max_junctions,
            feature_spatial_scale=feature_spatial_scale,
            junction_heatmap_criterion="binary_cross_entropy",
            # junction pooling parameters
            junction_pooling_size=junction_pooling_size,
            # adjacency matrix inference parameters
            block_inference_size=64,
            adjacency_matrix_criterion="binary_cross_entropy",
            weight_fn=None,
            is_train_junc=True,
            is_train_adj=True,
            enable_junc_infer=True,
            enable_adj_infer=True,
            verbose=True,
        )

    def forward(self, img):
        img = self.prep_data(img)
        feat = self.backbone(img)
        bs = img.size(0)

        junc_hm, junc_coords = self.junc_infer(feat)

        # first count number of detected junctions of each image
        junc_cnt = []
        j0 = 0
        for b in range(bs):
            junc_cnt.append(0)
            for j in range(j0, len(junc_coords)):
                if np.isclose(junc_coords[j, 0].item(), b, atol=.1):
                    junc_cnt[-1] += 1
                else:
                    j0 = j
                    break
        junc_st = np.cumsum([0] + junc_cnt).tolist()
        junc_pred = junc_coords.new_full((bs, self.max_junctions, 2), 0.)
        for b in range(bs):
            junc_pred[b, :junc_cnt[b]] = junc_coords[junc_st[b]:junc_st[b + 1], 1:] + .1

        # then for each image, build list of subgraph that processes at most block_size junctions
        block_infer = []
        for b in range(bs):
            num_blocks = junc_cnt[b] // self.adj_block_size + (1 if junc_cnt[b] % self.adj_block_size else 0)
            for bst in range(num_blocks):
                for bed in range(num_blocks):
                    st_st = junc_st[b] + bst * self.adj_block_size
                    st_len = min(self.adj_block_size, junc_cnt[b] - bst * self.adj_block_size)
                    ed_st = junc_st[b] + bed * self.adj_block_size
                    ed_len = min(self.adj_block_size, junc_cnt[b] - bed * self.adj_block_size)
                    block_infer.append(
                        BlockAdjacencyMatrixInference(
                            self.line_pool, self.adj_infer,
                            b, st_st, st_len, ed_st, ed_len, junc_coords,
                        )
                    )

        def output_collect_fn(outputs):
            output = img.new_full((bs, self.max_junctions, self.max_junctions), 0.)
            current_block = 0
            for b in range(bs):
                num_blocks = junc_cnt[b] // self.adj_block_size + (1 if junc_cnt[b] % self.adj_block_size else 0)
                for bst in range(num_blocks):
                    for bed in range(num_blocks):
                        st_st = bst * self.adj_block_size
                        st_len = min(self.adj_block_size, junc_cnt[b] - bst * self.adj_block_size)
                        ed_st = bed * self.adj_block_size
                        ed_len = min(self.adj_block_size, junc_cnt[b] - bed * self.adj_block_size)
                        output[b, st_st:st_st + st_len, ed_st:ed_st + ed_len] = outputs[current_block]
                        current_block += 1

            return output

        block_adj_infer = models.common.GradAccumulator(None, block_infer, output_collect_fn)
        feat_adj = self.adj_embed(feat)
        adj_matrix_pred, _ = block_adj_infer(feat_adj)

        return junc_pred, junc_hm, adj_matrix_pred
