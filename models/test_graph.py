from unittest import TestCase
import torch as th
import models.graph as graph
import matplotlib.pyplot as plt


class TestJunctionInference(TestCase):
    def test_junction_inference_forward(self):
        junc_infer = graph.JunctionInference(256, 0.1, 512, 0.25, False)
        with th.no_grad():
            feat_map = th.rand(5, 256, 32, 64)
            junc_map, junc_coord = junc_infer(feat_map)
            self.assertTrue(junc_map.size(0) == 5)
            self.assertTupleEqual(junc_map.shape[1:], (128, 256))
            self.assertTrue(junc_coord.size(1) == 3)
            self.assertTrue((junc_coord[:, 1:] >= 0).all() and (junc_coord[:, 1] < 256).all() and (junc_coord[:, 2] < 128).all())


class TestJunctionPooling(TestCase):
    def test_junc_pooling_forward(self):
        junc_infer = graph.JunctionInference(256, 0.1, 512, 0.25, False)
        junc_pool = graph.JunctionPooling(5, 5, 0.25)
        with th.no_grad():
            feat_map = th.rand(5, 256, 32, 64)
            junc_map, junc_coord = junc_infer(feat_map)
            out = junc_pool(feat_map, junc_coord)
            self.assertTrue(out.size(0) == junc_coord.size(0))
            self.assertTrue(out.size(1) == feat_map.size(1))
            self.assertTrue(out.shape[2:] == (5, 5))


class TestDirectionalAttention(TestCase):
    def test_attention_forward(self):
        attn = graph.DirectionalAttention(
            15,
            attn_sigma_dir=3.1415926/300,
            attn_sigma_pos=2
        )
        junc = th.tensor([
            [0., 0.],
            [2., 0.],
            [1., 1.],
            [0., 1.]
        ])
        map_st2ed, map_ed2st = attn(junc, junc)
        self.assertFalse(th.isnan(map_st2ed).any() or th.isnan(map_ed2st).any())
        self.assertTupleEqual(map_st2ed.size(), map_ed2st.size())
        self.assertTupleEqual(map_st2ed.shape[2:], map_ed2st.shape[2:])
        self.assertTupleEqual(map_st2ed.shape[2:], (15, 15))
        self.assertTupleEqual(map_st2ed.shape[:2], (4, 4))
        attn_map_st2ed = map_st2ed.numpy()
        attn_map_ed2st = map_ed2st.numpy()
        figs, axes = plt.subplots(4, 4)
        for i in range(4):
            for j in range(4):
                axes[i, j].imshow(attn_map_ed2st[i, j])
        plt.show()
        map_ct = attn(junc)
        self.assertTupleEqual(map_ct.size(), (15, 15))


class TestAdjacencyMatrixInference(TestCase):
    def test_adjacency_matrix_inference_forward(self):
        attn = graph.DirectionalAttention(15)
        junc_st = th.rand(30, 2)
        junc_ed = th.rand(60, 2)
        attn_st2ed, attn_ed2st = attn(junc_st, junc_ed)
        attn_center = attn()
        adj_with_center = graph.AdjacencyMatrixInference(256, junc_align_size=15, align_center=True)
        adj_wo_center = graph.AdjacencyMatrixInference(256, junc_align_size=15, align_center=False)
        feat_start = th.rand(30, 128, 15, 15)
        feat_end = th.rand(60, 128, 15, 15)
        feat_center = th.rand(30, 60, 128, 15, 15)
        adjacency_matrix_with_center = adj_with_center(feat_start, feat_end, attn_st2ed, attn_ed2st, feat_center, attn_center)
        adjacency_matrix_wo_center = adj_wo_center(feat_start, feat_end, attn_st2ed, attn_ed2st)
        self.assertTrue(adjacency_matrix_with_center.size() == adjacency_matrix_wo_center.size() == (30, 60))
