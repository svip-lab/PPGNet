from unittest import TestCase
import numpy as np
import models.common as common
import torch as th
import torch.nn as nn
from itertools import chain
from libs.roi_align.modules.roi_align import RoIAlign


class TestRoiPooling(TestCase):
    def setUp(self):
        th.manual_seed(1234)
        self.test_data = dict(
            input=th.rand(2, 3, 9, 10),
            rois=th.tensor([
                [0, 0, 0, 9, 8], # whole feature map
                [1, 0, 0, 9, 8], # whole feature map
                [0, 0, 0, 0, 0], # top left pixel
                [0, 0, 0, 1, 1], # top left 2x2
                [0, 0, 0, -1, -1], # bottom right out of range
                [1, 9, 8, 9, 8], # bottom right pixel
                [1, -3, -5, 6, 7], # top left out of range
                [1, -3, -5, 10, 11],  # both corner out of range
                [0, 3, 2, 9, 8], # 7x7 roi
                [0, 3, 3, 8, 8], # 6x6 roi
                [1, 1, 1, 5, 5], # 5x5 roi
            ], dtype=th.float32)
        )

    def test_output_size_7x7(self):
        input, rois = self.test_data["input"], self.test_data["rois"]
        output7x7 = common.roi_pooling(
            input=input,
            rois=rois,
            size=(7, 7),
            spatial_scale=1.0
        )
        # output7x7 = RoIAlign(aligned_height=7, aligned_width=7, spatial_scale=1.)(input, rois)
        self.assertEqual(output7x7.size(0), rois.size(0), "output size dismatches rois size")
        self.assertEqual(input.size(1), output7x7.size(1), "output channel dismatch input channel")
        self.assertTupleEqual(output7x7.shape[2:], (7, 7), "output shape dismatch required shape")

    def test_output_size_5x5(self):
        input, rois = self.test_data["input"], self.test_data["rois"]
        output5x5 = common.roi_pooling(
            input=input,
            rois=rois,
            size=(5, 5),
            spatial_scale=1.0
        )
        # output5x5 = RoIAlign(aligned_height=5, aligned_width=5, spatial_scale=1.)(input, rois)
        self.assertTupleEqual(output5x5.shape[2:], (5, 5), "output shape dismatch required shape")

    def test_output_size_1x1(self):
        input, rois = self.test_data["input"], self.test_data["rois"]
        output1x1 = common.roi_pooling(
            input=input,
            rois=rois,
            size=(1, 1),
            spatial_scale=1.0
        )
        self.assertTupleEqual(output1x1.shape[2:], (1, 1), "output shape dismatch required shape")

    def test_output_value(self):
        input, rois = self.test_data["input"], self.test_data["rois"]
        # rois[2, 3:] -= 1
        output1x1 = common.roi_pooling(
            input=input,
            rois=rois,
            size=(1, 1),
            spatial_scale=1.0
        )
        # output1x1 = RoIAlign(aligned_height=1, aligned_width=1, spatial_scale=1.)(input, rois)
        output5x5 = common.roi_pooling(
            input=input,
            rois=rois,
            size=(5, 5),
            spatial_scale=1.0
        )
        # output7x7 = common.roi_pooling(
        #     input=input,
        #     rois=rois,
        #     size=(7, 7),
        #     spatial_scale=1.0
        # )
        rois[8, 3:] -= 1
        rois[10, 3:] -= 1
        output7x7 = RoIAlign(aligned_height=7, aligned_width=7, spatial_scale=1.)(input, rois)
        output5x5 = RoIAlign(aligned_height=5, aligned_width=5, spatial_scale=1.)(input, rois)
        self.assertTrue((output1x1[2] == input[0, :, :1, :1]).all())
        self.assertTrue((output1x1[5] == input[1, :, 8:, 9:]).all())
        self.assertTrue((output5x5[10] == input[1, :, 1:6, 1:6]).all())
        self.assertTrue((output7x7[8] == input[0, :, 2:9, 3:10]).all())


class TestGradAccumulator(TestCase):
    def setUp(self):
        th.manual_seed(789)
        self.net1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1, bias=True),
        )
        self.test_data = dict(
            X = th.rand(100, 16, 32, 32),
            Y = th.rand(100, 1, 32, 32),
        )
        self.net1.apply(common.weights_init)
        self.net2.apply(common.weights_init)
        self.crit = nn.SmoothL1Loss()

    def test_forward_pass(self):
        X, Y = self.test_data["X"], self.test_data["Y"]
        with th.no_grad():
            feat = self.net1(X)
            out = []
            for i in range(4):
                for j in range(4):
                    out.append(self.net2(feat[:, :, i*4:(i+1)*4, j*4:(j+1)*4]))
            loss = 0
            for i in range(4):
                for j in range(4):
                    loss += self.crit(out[i*4+j], Y[:, :, i*4:(i+1)*4, j*4:(j+1)*4])
            loss /= 16

            class Net2(nn.Module):
                def __init__(self, net2, st_st, st_ed, ed_st, ed_ed):
                    super(Net2, self).__init__()
                    self.net2 = net2
                    self.st_st = st_st
                    self.st_ed = st_ed
                    self.ed_st = ed_st
                    self.ed_ed = ed_ed

                def forward(self, input):
                    return self.net2(input[:, :, self.st_st:self.st_ed, self.ed_st:self.ed_ed])

            class Crit(nn.Module):
                def __init__(self, crit, target, st_st, st_ed, ed_st, ed_ed):
                    super(Crit, self).__init__()
                    self.crit = crit
                    self.st_st = st_st
                    self.st_ed = st_ed
                    self.ed_st = ed_st
                    self.ed_ed = ed_ed
                    self.register_buffer("target", target)

                def forward(self, x):
                    return self.crit(x, self.target[:, :, self.st_st:self.st_ed, self.ed_st:self.ed_ed])

            gradacc = common.GradAccumulator(
                [Crit(self.crit, Y, i*4, (i+1)*4, j*4, (j+1)*4) for i in range(4) for j in range(4)],
                [Net2(self.net2, i*4, (i+1)*4, j*4, (j+1)*4) for i in range(4) for j in range(4)],
                collect_fn=None
            )
            net_ = nn.Sequential(self.net1, gradacc)
            out_, loss_ = net_(X)

        for i in range(len(out)):
            self.assertTrue(th.allclose(out[i], out_[i]))
        self.assertTrue(th.allclose(loss, loss_), f"{loss}\n{loss_}")

    def test_backward_pass(self):
        X, Y = self.test_data["X"], self.test_data["Y"]

        self.net1.zero_grad()
        self.net2.zero_grad()

        feat = self.net1(X)
        out = []
        for i in range(4):
            for j in range(4):
                out.append(self.net2(feat[:, :, i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]))
        loss = 0
        for i in range(4):
            for j in range(4):
                loss += self.crit(out[i * 4 + j], Y[:, :, i * 4:(i + 1) * 4, j * 4:(j + 1) * 4])
        loss /= 16
        loss.backward()

        grad = {}

        for k, v in chain(self.net1.named_parameters(prefix="net1"), self.net2.named_parameters(prefix="net2")):
            grad[k] = th.tensor(v.grad)

        self.net1.zero_grad()
        self.net2.zero_grad()

        class Net2(nn.Module):
            def __init__(self, net2, st_st, st_ed, ed_st, ed_ed):
                super(Net2, self).__init__()
                self.net2 = net2
                self.st_st = st_st
                self.st_ed = st_ed
                self.ed_st = ed_st
                self.ed_ed = ed_ed

            def forward(self, input):
                return self.net2(input[:, :, self.st_st:self.st_ed, self.ed_st:self.ed_ed])

        class Crit(nn.Module):
            def __init__(self, crit, target, st_st, st_ed, ed_st, ed_ed):
                super(Crit, self).__init__()
                self.crit = crit
                self.st_st = st_st
                self.st_ed = st_ed
                self.ed_st = ed_st
                self.ed_ed = ed_ed
                self.register_buffer("target", target)

            def forward(self, x):
                return self.crit(x, self.target[:, :, self.st_st:self.st_ed, self.ed_st:self.ed_ed])

        gradacc = common.GradAccumulator(
            [Crit(self.crit, Y, i * 4, (i + 1) * 4, j * 4, (j + 1) * 4) for i in range(4) for j in range(4)],
            [Net2(self.net2, i * 4, (i + 1) * 4, j * 4, (j + 1) * 4) for i in range(4) for j in range(4)],
            collect_fn=None
        )
        net_ = nn.Sequential(self.net1, gradacc)
        out_, loss_ = net_(X)
        loss_.backward()

        grad_ = {}
        for k, v in chain(self.net1.named_parameters(prefix="net1"), self.net2.named_parameters(prefix="net2")):
            grad_[k] = th.tensor(v.grad)

        for k in sorted(grad.keys()):
            self.assertTrue(th.allclose(grad[k], grad_[k]), f"{k}:\n{grad[k]}\n{grad_[k]}")
