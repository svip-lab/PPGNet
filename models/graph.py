import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import conv3x3_bn_relu, LMFPeakFinder, weights_init


class JunctionInference(nn.Module):
    def __init__(self, dim_embedding, pooling_threshold=0.2, max_junctions=512, spatial_scale=0.25, verbose=False):
        super(JunctionInference, self).__init__()
        self.dim_embedding = dim_embedding
        self.pool_th = pooling_threshold
        self.max_juncs = max_junctions
        self.map_infer = nn.Sequential(
            conv3x3_bn_relu(dim_embedding, dim_embedding // 2, 1),
            conv3x3_bn_relu(dim_embedding // 2, dim_embedding // 2, 1),
            nn.Conv2d(dim_embedding // 2, 1, 1),
            nn.Sigmoid()
        )
        self.verbose = verbose
        self.scale = spatial_scale
        self.map_infer.apply(weights_init)

    def forward(self, feat):
        bs, ch, h, w = feat.size()
        junc_map = self.map_infer(feat)
        junc_map = nn.functional.interpolate(
            junc_map,
            scale_factor=1. / self.scale,
            mode="bilinear",
            align_corners=False
        )
        junc_coord = []
        for b in range(bs):
            peak_finder = LMFPeakFinder(min_th=self.pool_th)
            coord, score = peak_finder.detect(junc_map[b, 0].data.cpu().numpy())
            if self.verbose:
                print(f"find {len(coord)} jucntions.", flush=True)
            if coord is None or len(coord) == 0:
                continue
            junc_score = torch.from_numpy(score).to(feat)
            _, ind = torch.sort(junc_score, descending=True)
            ind = ind.cpu() 
            coord = coord[ind[:self.max_juncs]]
            coord = coord.reshape((-1, 2))
            y, x = coord[:, 0], coord[:, 1]
            y = torch.from_numpy(y).to(feat)
            x = torch.from_numpy(x).to(feat)
            assert (x >= 0).all() and (x < junc_map.size(3)).all() and (y >= 0).all() and (y < junc_map.size(2)).all()
            junc_coord.append(
                torch.cat([feat.new_full((len(x), 1), b), x.view(-1, 1), y.view(-1, 1)], dim=1)
            )
        if len(junc_coord) > 0:
            junc_coord = torch.cat(junc_coord, dim=0)
        else:
            junc_coord = feat.new_full((1, 3), 0.)

        return junc_map.squeeze(1), junc_coord


class LinePooling(nn.Module):
    def __init__(self, align_size=256, spatial_scale=0.25):
        super(LinePooling, self).__init__()
        self.align_size = align_size
        assert isinstance(self.align_size, int)
        self.scale = spatial_scale

    def forward(self, feat, coord_st, coord_ed):
        _, ch, h, w = feat.size()
        num_st, num_ed = coord_st.size(0), coord_ed.size(0)
        assert coord_st.size(1) == 3 and coord_ed.size(1) == 3
        assert (coord_st[:, 0] == coord_st[0, 0]).all() and (coord_ed[:, 0] == coord_st[0, 0]).all()
        bs = coord_st[0, 0].item()
        # construct bounding boxes from junction points
        with torch.no_grad():
            coord_st = coord_st[:, 1:] * self.scale
            coord_ed = coord_ed[:, 1:] * self.scale
            coord_st = coord_st.unsqueeze(1).expand(num_st, num_ed, 2)
            coord_ed = coord_ed.unsqueeze(0).expand(num_st, num_ed, 2)
            arr_st2ed = coord_ed - coord_st
            sample_grid = torch.linspace(0, 1, steps=self.align_size).to(feat).view(1, 1, self.align_size).expand(num_st, num_ed, self.align_size)
            sample_grid = torch.einsum("ijd,ijs->ijsd", (arr_st2ed, sample_grid)) + coord_st.view(num_st, num_ed, 1, 2).expand(num_st, num_ed, self.align_size, 2)
            sample_grid = sample_grid.view(num_st, num_ed, self.align_size, 2)
            sample_grid[..., 0] = sample_grid[..., 0] / (w - 1) * 2 - 1
            sample_grid[..., 1] = sample_grid[..., 1] / (h - 1) * 2 - 1

        output = F.grid_sample(feat[int(bs)].view(1, ch, h, w).expand(num_st, ch, h, w), sample_grid)
        assert output.size() == (num_st, ch, num_ed, self.align_size)
        output = output.permute(0, 2, 1, 3).contiguous()

        return output


class AdjacencyMatrixInference(nn.Module):
    def __init__(self, dim_embedding=256, align_size=256):
        super(AdjacencyMatrixInference, self).__init__()
        self.dim_embedding = dim_embedding
        self.align_size = align_size
        self.dblock = nn.Sequential(
            nn.Conv1d(dim_embedding, dim_embedding, 8, 4, 2, bias=False),
            nn.GroupNorm(32, dim_embedding),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_embedding, dim_embedding, 8, 4, 2, bias=False),
            nn.GroupNorm(32, dim_embedding),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim_embedding, dim_embedding, 8, 4, 2, bias=False),
            nn.GroupNorm(32, dim_embedding),
            nn.ReLU(inplace=True)
        )
        self.connectivity_inference = nn.Sequential(
            nn.Conv1d(dim_embedding, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, line_feat):
        num_st, num_ed, c, s = line_feat.size()
        output_st2ed = line_feat.view(num_st * num_ed, c, s)
        output_ed2st = torch.flip(output_st2ed, (2, ))
        output_st2ed = self.dblock(output_st2ed)
        output_ed2st = self.dblock(output_ed2st)
        adjacency_matrix1 = self.connectivity_inference(output_st2ed).view(num_st, num_ed)
        adjacency_matrix2 = self.connectivity_inference(output_ed2st).view(num_st, num_ed)

        return torch.min(adjacency_matrix1, adjacency_matrix2)
