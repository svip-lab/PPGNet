from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.cluster.hierarchy import fclusterdata
import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from collections import Sized, Iterable


class LMFPeakFinder(object):
    """
    shamelessly borrow from https://stackoverflow.com/a/3689710
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    def __init__(self, min_dist=5., min_th=0.3):
        self.min_dist = min_dist
        self.min_th = min_th

    def detect(self, image):
        # define an 8-connected neighborhood
        neighborhood = generate_binary_structure(2, 2)

        # apply the local maximum filter; all pixel of maximal value
        # in their neighborhood are set to 1
        local_max = maximum_filter(image, footprint=neighborhood) == image
        # local_max is a mask that contains the peaks we are
        # looking for, but also the background.
        # In order to isolate the peaks we must remove the background from the mask.

        # we create the mask of the background
        background = (image < self.min_th)

        # a little technicality: we must erode the background in order to
        # successfully subtract it form local_max, otherwise a line will
        # appear along the background border (artifact of the local maximum filter)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

        # we obtain the final mask, containing only peaks,
        # by removing the background from the local_max mask (xor operation)
        detected_peaks = local_max ^ eroded_background

        detected_peaks[image < self.min_th] = False
        peaks = np.array(np.nonzero(detected_peaks)).T

        if len(peaks) == 0:
            return peaks, np.array([])

        # nms
        if len(peaks) == 1:
            clusters = [0]
        else:
            clusters = fclusterdata(peaks, self.min_dist, criterion="distance")
        peak_groups = {}
        for ind_junc, ind_group in enumerate(clusters):
            if ind_group not in peak_groups.keys():
                peak_groups[ind_group] = []
                peak_groups[ind_group].append(peaks[ind_junc])
        peaks_nms = []
        peaks_score = []
        for peak_group in peak_groups.values():
            values = [image[y, x] for y, x in peak_group]
            ind_max = np.argmax(values)
            peaks_nms.append(peak_group[int(ind_max)])
            peaks_score.append(values[int(ind_max)])

        return np.float32(np.array(peaks_nms)), np.float32(np.array(peaks_score))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(1e-4)


def roi_pooling(input, rois, size=(7, 7), spatial_scale=1.0):
    assert (rois.dim() == 2)
    assert (rois.size(1) == 5)
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)

    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        if roi[1] >= input.size(3) or roi[2] >= input.size(2) or roi[1] < 0 or roi[2] < 0:
            # print(f"Runtime Warning: roi top left corner out of range: {roi}", file=sys.stderr)
            roi[1] = torch.clamp(roi[1], 0, input.size(3) - 1)
            roi[2] = torch.clamp(roi[2], 0, input.size(2) - 1)
        if roi[3] >= input.size(3) or roi[4] >= input.size(2) or roi[3] < 0 or roi[4] < 0:
            # print(f"Runtime Warning: roi bottom right corner out of range: {roi}", file=sys.stderr)
            roi[3] = torch.clamp(roi[3], 0, input.size(3) - 1)
            roi[4] = torch.clamp(roi[4], 0, input.size(2) - 1)
        if (roi[3:5] - roi[1:3] < 0).any():
            # print(f"Runtime Warning: invalid roi: {roi}", file=sys.stderr)
            im = input.new_full((1, input.size(1), 1, 1), 0)
        else:
            im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
        output.append(F.adaptive_max_pool2d(im, size))

    return torch.cat(output, 0)


class GradAccumulatorFunction(Function):
    @staticmethod
    def forward(ctx, input, accumulated_grad=None, mode="release"):
        ctx.accumulated_grad = accumulated_grad
        ctx.mode = mode
        return input

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        accumulated_grad = ctx.accumulated_grad
        ctx.accumulated_grad = None
        if ctx.mode == "accumulate":
            accumulated_grad.add_(grad_output)
            return torch.zeros_like(grad_output), None, None
        elif ctx.mode == "release":
            if accumulated_grad is not None:
                accumulated_grad.add_(grad_output)
            else:
                accumulated_grad = grad_output
            grad_output = accumulated_grad
            return grad_output, None, None
        else:
            raise ValueError(f"invalid mode {ctx.mode}")


class GradAccumulator(nn.Module):
    """
    Helper module used to accumulate gradient of the given tensor w.r.t output of criterion.
    Typically used when we have a feature extractor followed by several modules that can be calculate independently, the
    module only retains the last executed submodule and accumulate the gradient produced by former submodules, so that
    GPU memory used to store the temporary variables in former submodules is saved. It can be also used to extend
    effective batch size at little expense of memory.
    """
    def __init__(self, criterion_fns, submodules, collect_fn=None, reduce_method="mean"):
        super(GradAccumulator, self).__init__()
        assert isinstance(submodules, (Sized, Iterable)), "invalid submodules"
        if isinstance(criterion_fns, (Sized, Iterable)):
            assert len(submodules) == len(criterion_fns)
            assert all([isinstance(submodule, nn.Module) for submodule in submodules])
            assert all([isinstance(criterion_fn, nn.Module) for criterion_fn in criterion_fns])
        elif isinstance(criterion_fns, nn.Module):
            criterion_fns = [criterion_fns for _ in range(len(submodules))]
        elif criterion_fns is None:
            criterion_fns = [criterion_fns for _ in range(len(submodules))]
        else:
            raise ValueError("invalid criterion function")
        assert reduce_method in ("mean", "sum", None)

        self.submodules = nn.ModuleList(submodules)
        self.criterion_fns = nn.ModuleList(criterion_fns)
        self.method = reduce_method
        self.grad_buffer = None
        self.func = GradAccumulatorFunction.apply
        self.collect_fn = collect_fn

    def forward(self, tensor):
        outputs = []
        losses = tensor.new_full((1,), 0)
        self.grad_buffer = None
        for i, (submodule, criterion) in enumerate(zip(self.submodules, self.criterion_fns)):
            mode = "accumulate" if i < len(self.submodules) - 1 else "release"
            if self.grad_buffer is None:
                self.grad_buffer = torch.zeros_like(tensor)
            if mode == "accumulate":
                output = tensor.detach()
                output.requires_grad = True
            else:
                output = tensor
            output = self.func(
                output,
                self.grad_buffer,
                mode,
            )
            if isinstance(output, tuple):
                output = submodule(*output)
            else:
                output = submodule(output)
            if criterion is not None:
                loss = criterion(output)
                if self.method == "mean":
                    loss = loss / len(self.submodules)

                if mode == "accumulate" and torch.is_grad_enabled():
                    loss.backward()
                    loss = loss.detach()

                output = output.detach()
                losses += loss
            else:
                assert not output.requires_grad, "criterion must be specified to calculate output gradient"

            outputs.append(output)

        if self.collect_fn is not None:
            with torch.no_grad():
                outputs = self.collect_fn(outputs)

        return outputs, losses
