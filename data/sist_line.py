import os
import numpy as np
import torch as th
from torch.utils import data
from data.line_graph import LineGraph
from glob import glob
from PIL import Image
from data.utils import gen_gaussian_map


class SISTLine(data.Dataset):
    def __init__(self, data_root, transforms, phase="train", sigma_junction=3., max_junctions=512):
        self.data_root = data_root
        self.img = [os.path.join(phase, os.path.basename(f)) for f in glob(os.path.join(data_root, phase, "*.jpg"))]
        self.transforms = transforms
        self.phase = phase
        self.max_junctions = max_junctions
        self.sigma_junction = sigma_junction

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.data_root, self.img[item]))
        ori_w, ori_h = img.size
        
        lg = LineGraph().load(os.path.join(self.data_root, self.img[item][:-4] + ".lg"))
        num_junc = lg.num_junctions
        # assert num_junc <= self.max_junctions, f"{(item, num_junc)}"
        if self.phase == "train" and num_junc > self.max_junctions:
            return self[(item + 1) % len(self)]
        elif num_junc > self.max_junctions:
            return self[(item + 1) % len(self)]
            # raise AssertionError()
        junc = np.zeros((self.max_junctions, 2))
        # tic = time()
        junc[:num_junc] = np.array([j if np.sum(j) > 0 else j + 1 for j in lg.junctions()])
        # print(f"junc time: {time() - tic:.4f}")

        assert np.sum(junc[:num_junc].sum(axis=1) <= 0) == 0, f"{item}"
        # tic = time()
        adj_mtx = np.zeros((self.max_junctions, self.max_junctions))
        # print(f"mtx time: {time() - tic:.4f}")
        adj_mtx[:num_junc, :num_junc] = lg.adj_mtx

        if self.transforms is not None:
            img, junc = self.transforms(img, junc)

        cur_w, cur_h = img.size

        junc[junc >= img.size[0]] = img.size[0] - 1
        junc[junc < 0] = 0
        # tic = time()
        heatmap = gen_gaussian_map(junc[:num_junc], img.size[:2], self.sigma_junction)
        assert cur_h == cur_w
        line_map = lg.line_map(cur_h, cur_w / ori_w, cur_h / ori_h, line_width=self.sigma_junction)
        # print(f"gaussian time: {time() - tic:.4f}")

        img = np.array(np.asarray(img)[:, :, ::-1])
        img = th.from_numpy(img).permute(2, 0, 1)
        adj_mtx = th.from_numpy(adj_mtx)
        junc = th.from_numpy(junc)
        heatmap = th.from_numpy(heatmap)
        line_map = th.from_numpy(line_map)

        batch = dict(
            image=img.float(),
            adj_mtx=adj_mtx.float(),
            heatmap = heatmap.float(),
            junctions = junc.float(),
            line_map = line_map.float()
        )

        return batch

    def __call__(self, item):
        return self.__getitem__(item)

    def __len__(self):
        return len(self.img)


if __name__ == "__main__":
    from tqdm import trange
    from multiprocessing.pool import Pool
    data = SISTLine("/home/ziheng/indoorDist_new", None, "train")
    # os.makedirs("/home/ziheng/heatmaps")
    pool = Pool(20)
    cnt = 0

    def readnsave(i):
        batch = data[i]
        hm = batch["heatmap"].numpy()
        np.save(f"/home/ziheng/heatmaps/{i}.npy", hm)

    def juncsave(i):
        batch = data[i]
        hm = batch["heatmap"].numpy()
        junc = batch["heatmap"].numpy()
        np.save(f"/home/ziheng/heatmaps/{i}.npy", hm)


    readnsave.cnt = 0
    # for i in trange(len(data)):
    pool.map_async(readnsave, range(len(data)))
    pool.close()
    pool.join()

