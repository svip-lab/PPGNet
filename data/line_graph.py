import os
from sklearn.neighbors import KDTree
from scipy.cluster.hierarchy import fclusterdata
import pickle
from itertools import combinations
import collections
from data.common import *
import cv2


class LineGraph(object):
    def __init__(
            self, eps_junction=3., eps_line_deg=np.pi / 20, verbose=False
    ):
        self._line_segs = []
        self._junctions = []
        self._end_points = []
        self._refine_junctions = None
        self._neighbor = {}
        self._junc2line = {}
        self._freeze_junction = False
        self._kdtree = None
        self._eps_junc = eps_junction
        self._eps_line_seg = eps_line_deg
        self.verbose = verbose

    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            for mem in dir(self):
                if (
                        mem.startswith("_")
                        and not mem.startswith("__")
                        and not isinstance(getattr(self, mem), collections.Callable)
                ):
                    setattr(self, mem, data[mem])
        return self

    def save(self, filename):
        with open(filename, "wb+") as f:
            data = {}
            for mem in dir(self):
                if (
                        mem.startswith("_")
                        and not mem.startswith("__")
                        and not isinstance(getattr(self, mem), collections.Callable)
                ):
                    data[mem] = getattr(self, mem)
            pickle.dump(data, f)
        return self

    def _is_pt_in_line_seg(self, pt, pt1, pt2):

        return is_pt_in_line_seg(self._eps_junc, tuple(pt), tuple(pt1), tuple(pt2))
        # param, _ = fit_line([pt1, pt2])
        # _, alphas = project_pts_on_line([pt, pt1, pt2], param)
        # dist = dist_pts_to_line([pt], param)[0]
        # return (dist < self._eps_junc * 2) and (np.min(alphas[1:]) <= alphas[0] <= np.max(alphas[1:]))

    def freeze_junction(self, status=True):
        self._freeze_junction = status
        if status:
            clusters = fclusterdata(self._junctions, self._eps_junc, criterion="distance")
            junc_groups = {}
            for ind_junc, ind_group in enumerate(clusters):
                if ind_group not in junc_groups.keys():
                    junc_groups[ind_group] = []
                junc_groups[ind_group].append(self._junctions[ind_junc])
            if self.verbose:
                print(f"{len(self._junctions) - len(junc_groups)} junctions merged.")
            self._junctions = [np.mean(junc_group, axis=0) for junc_group in junc_groups.values()]

            self._kdtree = KDTree(self._junctions, leaf_size=30)
            dists, inds = self._kdtree.query(self._junctions, k=2)
            repl_inds = np.nonzero(dists.sum(axis=1) < self._eps_junc)[0].tolist()
            # assert len(repl_inds) == 0
        else:
            self._kdtree = None

    def _can_be_extented_by(self, line_seg1, line_seg2):
        pt1 = line_seg1["pt1"]
        pt2 = line_seg1["pt2"]
        _pt1 = line_seg2["pt1"]
        _pt2 = line_seg2["pt2"]
        if self._is_pt_in_line_seg(_pt1, pt1, pt2) or self._is_pt_in_line_seg(_pt2, pt1, pt2):
            arr1 = pt1 - pt2
            arr2 = _pt1 - _pt2
            arr1 /= np.linalg.norm(arr1)
            arr2 /= np.linalg.norm(arr2)
            if np.abs(arr1.dot(arr2)) > np.cos(self._eps_line_seg):
                return True

    def freeze_line_seg(self, status=True):
        assert self._freeze_junction, "junction should be freezed before."
        self._freeze_line_seg = status
        if status:
            # remove all junctions that are not in any line segment
            junc_remove = set(list(range(len(self._junctions))))
            for line_seg in self._line_segs:
                junc_remove -= line_seg["junctions"]

            junc_remain = [ind_junc for ind_junc in range(len(self._junctions)) if ind_junc not in junc_remove]
            self._junctions = [junc for ind_junc, junc in enumerate(self._junctions) if ind_junc not in junc_remove]
            map_old_to_new = {old: new for new, old in enumerate(junc_remain)}
            for line_seg in self._line_segs:
                line_seg["junctions"] = set([map_old_to_new[old] for old in line_seg["junctions"]])
            self._junc2line = {}

            # # extend all line segments
            # cnt = 0
            # finished = False
            # while (not finished):
            #     finished = True
            #     for ind_ls1 in range(len(self._line_segs)):
            #         for ind_ls2 in range(ind_ls1 + 1, len(self._line_segs)):
            #             # if ind_ls1 == ind_ls2:
            #             #     continue
            #             ls1 = self._line_segs[ind_ls1]
            #             ls2 = self._line_segs[ind_ls2]
            #             if ls2["junctions"].issubset(ls1["junctions"]):
            #                 continue
            #             if self._can_be_extented_by(ls1, ls2):
            #                 pt1 = ls1["pt1"]
            #                 pt2 = ls1["pt2"]
            #                 param = ls1["param"]
            #                 _pt1 = ls2["pt1"]
            #                 _pt2 = ls2["pt2"]
            #                 _param = ls2["param"]
            #                 P, alphas = project_pts_on_line([pt1, pt2, _pt1, _pt2], param)
            #                 _P, _alphas = project_pts_on_line([pt1, pt2, _pt1, _pt2], _param)
            #                 ind_min, ind_max = np.argmin(alphas), np.argmax(alphas)
            #                 _ind_min, _ind_max = np.argmin(_alphas), np.argmax(_alphas)
            #                 if np.abs(alphas[ind_min] - alphas[ind_max]) < self._eps_junc:  # this should not happen...
            #                     continue
            #                 ls1["pt1"] = P[ind_min]
            #                 ls1["pt2"] = P[ind_max]
            #                 ls2["pt1"] = _P[_ind_min]
            #                 ls2["pt2"] = _P[_ind_max]
            #                 ls1["junctions"] = ls1["junctions"].union(ls2["junctions"])
            #                 ls2["junctions"] = ls2["junctions"].union(ls1["junctions"])
            #                 cnt += 1
            #                 finished = False
            #
            # if self.verbose:
            #     print(f"line segments extend {cnt} times.", flush=True)
            #
            # # merge line segments of which junction set is subset of that of other line segments
            # finished = False
            # merged_inds = []
            # while (not finished):
            #     finished = True
            #     for ind_ls1 in range(len(self._line_segs)):
            #         for ind_ls2 in range(len(self._line_segs)):
            #             if (ind_ls1 == ind_ls2) or (ind_ls2 in merged_inds):
            #                 continue
            #             ls1 = self._line_segs[ind_ls1]
            #             ls2 = self._line_segs[ind_ls2]
            #             if ls2["junctions"].issubset(ls1["junctions"]):
            #                 merged_inds.append(ind_ls2)
            #                 finished = False
            # self._line_segs = [line_seg for ind, line_seg in enumerate(self._line_segs) if ind not in merged_inds]
            # if self.verbose:
            #     print(f"{len(merged_inds)} line segments merged.", flush=True)

            # # refine line segments w.r.t all associated junctions
            # ling_seg_remove = []
            # for ind_line_seg, line_seg in enumerate(self._line_segs):
            #     junc_set = line_seg["junctions"]
            #     if len(junc_set) == 2:  # no need to refine
            #         continue
            #     ind_innr_junctions = [ind_junc for ind_junc in junc_set]
            #     param, res = fit_line([self._junctions[ind_junc] for ind_junc in ind_innr_junctions])
            #     ind_outliers = np.nonzero(res > self._eps_junc * 2)[0]
            #     # if too many outliers, remove this line segment
            #     if (len(junc_set) - len(ind_outliers) < 2) or (len(ind_outliers) / len(junc_set) > 0.3):
            #         # ling_seg_remove.append(ind_line_seg)
            #         continue
            #     # else remove outliers
            #     for ind_outlier in ind_outliers:
            #         # junc_set.remove(ind_innr_junctions[ind_outlier])
            #         pass
            #     line_seg["param"] = param
            #     # find new endpoints
            #     ind_innr_junctions = [ind_junc for ind_junc in junc_set]
            #     P, alphas = project_pts_on_line([self._junctions[ind_junc] for ind_junc in ind_innr_junctions], param)
            #     ind_max, ind_min = np.argmax(alphas), np.argmin(alphas)
            #     line_seg["pt1"] = P[ind_min]
            #     line_seg["pt2"] = P[ind_max]
            #
            # self._line_segs = [line_seg for ind, line_seg in enumerate(self._line_segs) if ind not in ling_seg_remove]
            # if self.verbose:
            #     print(f"{len(ling_seg_remove)} line segments removed.", flush=True)

            # # remove all junctions that are not in any line segment
            # junc_remove = set(list(range(len(self._junctions))))
            # for line_seg in self._line_segs:
            #     junc_remove -= line_seg["junctions"]
            # junc_remain = [ind_junc for ind_junc in range(len(self._junctions)) if ind_junc not in junc_remove]
            # self._junctions = [junc for ind_junc, junc in enumerate(self._junctions) if ind_junc not in junc_remove]
            # map_old_to_new = {old: new for new, old in enumerate(junc_remain)}
            # for line_seg in self._line_segs:
            #     line_seg["junctions"] = set([map_old_to_new[old] for old in line_seg["junctions"]])
            # self._junc2line = {}
            #
            # # build hash table mapping from junction id to line segment
            # for line_seg in self._line_segs:
            #     for ind_junc in line_seg["junctions"]:
            #         if ind_junc not in self._junc2line.keys():
            #             self._junc2line[ind_junc] = []
            #         self._junc2line[ind_junc].append(line_seg)

            # # remove possibly noise line segments
            # line_seg_remove = []
            # for ind_line_seg, line_seg in enumerate(self._line_segs):
            #     junc_set = line_seg["junctions"]
            #     if len(junc_set) == 1: # not possible
            #         line_seg_remove.append(ind_line_seg)
            #     elif len(junc_set) == 2:
            #         ind_junc1, ind_junc2 = list(junc_set)
            #         if len(self._junc2line[ind_junc1]) >= 5 or len(self._junc2line[ind_junc2]) >= 5:
            #             is_delete = True
            #             for neigh_line_seg in self._junc2line[ind_junc1]:
            #                 if neigh_line_seg is line_seg:
            #                     continue
            #                 arr1 = neigh_line_seg["pt1"] - neigh_line_seg["pt2"]
            #                 arr2 = line_seg["pt1"] - line_seg["pt2"]
            #                 if np.abs(arr1.dot(arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))) < np.cos(
            #                         self._eps_line_seg):
            #                     for neigh_line_seg2 in self._junc2line[ind_junc2]:
            #                         if neigh_line_seg2 is line_seg:
            #                             continue
            #                         arr1 = neigh_line_seg2["pt1"] - neigh_line_seg2["pt2"]
            #                         arr2 = line_seg["pt1"] - line_seg["pt2"]
            #                         if np.abs(arr1.dot(arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))) < np.cos(
            #                                 self._eps_line_seg):
            #                             is_delete = False
            #                             break
            #             if is_delete:
            #                 line_seg_remove.append(ind_line_seg)
            #                 break
            #         if len(self._junc2line[ind_junc1]) == 1 or len(self._junc2line[ind_junc1]) == 1:
            #             for neigh_line_seg in self._junc2line[ind_junc1]:
            #                 if neigh_line_seg is line_seg:
            #                     continue
            #                 arr1 = neigh_line_seg["pt1"] - neigh_line_seg["pt2"]
            #                 arr2 = line_seg["pt1"] - line_seg["pt2"]
            #                 if np.abs(arr1.dot(arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))) < np.cos(self._eps_line_seg):
            #                     line_seg_remove.append(ind_line_seg)
            #                     break
            #             for neigh_line_seg in self._junc2line[ind_junc2]:
            #                 if neigh_line_seg is line_seg:
            #                     continue
            #                 arr1 = neigh_line_seg["pt1"] - neigh_line_seg["pt2"]
            #                 arr2 = line_seg["pt1"] - line_seg["pt2"]
            #                 if np.abs(arr1.dot(arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))) < np.cos(self._eps_line_seg):
            #                     line_seg_remove.append(ind_line_seg)
            #                     break
            # self._line_segs = [line_seg for ind, line_seg in enumerate(self._line_segs) if ind not in line_seg_remove]
            # if self.verbose:
            #     print(f"{len(line_seg_remove)} possibly noisy line segments removed.", flush=True)

            # # remove all junctions that are not in any line segment
            # junc_remove = set(list(range(len(self._junctions))))
            # for line_seg in self._line_segs:
            #     junc_remove -= line_seg["junctions"]
            # junc_remain = [ind_junc for ind_junc in range(len(self._junctions)) if ind_junc not in junc_remove]
            # self._junctions = [junc for ind_junc, junc in enumerate(self._junctions) if ind_junc not in junc_remove]
            # map_old_to_new = {old: new for new, old in enumerate(junc_remain)}
            # for line_seg in self._line_segs:
            #     line_seg["junctions"] = set([map_old_to_new[old] for old in line_seg["junctions"]])
            # self._junc2line = {}

            # build hash table mapping from junction id to line segment
            for line_seg in self._line_segs:
                for ind_junc in line_seg["junctions"]:
                    if ind_junc not in self._junc2line.keys():
                        self._junc2line[ind_junc] = []
                    self._junc2line[ind_junc].append(line_seg)
            assert len(self._junc2line) == len(self._junctions)

            # # refine all junctions w.r.t associated line segments
            # junctions_refined = []
            # for ind_junc, junc in enumerate(self._junctions):
            #     line_segs = self._junc2line[ind_junc]
            #     # assert len(line_segs) > 0, "line seg"
            #     if len(line_segs) == 1:
            #         param = line_segs[0]["param"]
            #         refined, _ = project_pts_on_line([junc], param)
            #         junctions_refined.append(refined[0])
            #     elif len(line_segs) >= 2:
            #         refined, _ = find_lines_intersect([line_seg["param"] for line_seg in line_segs])
            #         if np.linalg.norm(refined - junc) > 3 * self._eps_junc:  # maybe something wrong, do nothing...
            #             junctions_refined.append(junc)
            #         junctions_refined.append(refined)
            # self._junctions = junctions_refined
            self._kdtree = KDTree(self._junctions, leaf_size=30)

            # # remove all junctions that are not in any line segment
            # junc_remove = set(list(range(len(self._junctions))))
            # for line_seg in self._line_segs:
            #     junc_remove -= line_seg["junctions"]
            # junc_remain = [ind_junc for ind_junc in range(len(self._junctions)) if ind_junc not in junc_remove]
            # self._junctions = [junc for ind_junc, junc in enumerate(self._junctions) if ind_junc not in junc_remove]
            # map_old_to_new = {old: new for new, old in enumerate(junc_remain)}
            # for line_seg in self._line_segs:
            #     line_seg["junctions"] = set([map_old_to_new[old] for old in line_seg["junctions"]])
            # self._junc2line = {}

            # # build hash table mapping from junction id to line segment
            # for line_seg in self._line_segs:
            #     for ind_junc in line_seg["junctions"]:
            #         if ind_junc not in self._junc2line.keys():
            #             self._junc2line[ind_junc] = []
            #         self._junc2line[ind_junc].append(line_seg)

            # add line segment intersections
            cnt_new = 0
            for ind_ls1 in range(len(self._line_segs)):
                for ind_ls2 in range(ind_ls1 + 1, len(self._line_segs)):
                    ls1, ls2 = self._line_segs[ind_ls1], self._line_segs[ind_ls2]
                    pt11, pt12 = ls1["pt1"], ls1["pt2"]
                    pt21, pt22 = ls2["pt1"], ls2["pt2"]
                    p = np.array(pt11)
                    r = np.array(pt12) - np.array(pt11)
                    q = np.array(pt21)
                    s = np.array(pt22) - np.array(pt21)
                    alpha = np.cross(r, s)
                    if np.isclose(alpha, 0):
                        continue
                    # if np.abs(np.dot(r, s) / np.linalg.norm(r) / np.linalg.norm(s)) > self._eps_line_seg:
                    #     continue
                    beta_t = np.cross(q - p, s)
                    beta_u = np.cross(q - p, r)
                    t = np.mean(beta_t / alpha)
                    u = np.mean(beta_u / alpha)
                    # exact intersect
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        # print("find exact intersect")
                        assert np.allclose(p + t * r, q + u * s, rtol=1.e-3), "intersecting math assertion (exact)"
                        intersect = p + t * r
                        dists, ind = self._kdtree.query([intersect], k=1)
                        if dists[0, 0] < self._eps_junc:
                            ls1["junctions"].add(ind[0, 0])
                            ls2["junctions"].add(ind[0, 0])
                            self._junc2line[ind[0, 0]].append(ls1)
                            self._junc2line[ind[0, 0]].append(ls2)
                        else:
                            ind = len(self._junctions)
                            self._junctions.append(intersect)
                            ls1["junctions"].add(ind)
                            ls2["junctions"].add(ind)
                            self._junc2line[ind] = [ls1, ls2]
                            self._kdtree = KDTree(self._junctions, leaf_size=30)
                            cnt_new += 1

                    # # close to intersect
                    # elif (min(abs(t), abs(t - 1)) * np.linalg.norm(r) < self._eps_junc * 5) and (
                    #         min(abs(u), abs(u - 1)) * np.linalg.norm(s) < self._eps_junc * 5):
                    #     assert np.allclose(p + t * r, q + u * s, rtol=1.e-3), "intersecting math assertion (close)"
                    #     intersect = p + t * r
                    #     dists, ind = self._kdtree.query([intersect], k=1)
                    #     if dists[0, 0] < self._eps_junc:
                    #         ls1["junctions"].add(ind[0, 0])
                    #         ls2["junctions"].add(ind[0, 0])
                    #         self._junc2line[ind[0, 0]].append(ls1)
                    #         self._junc2line[ind[0, 0]].append(ls2)
                    #     else:
                    #         ind = len(self._junctions)
                    #         self._junctions.append(intersect)
                    #         ls1["junctions"].add(ind)
                    #         ls2["junctions"].add(ind)
                    #         self._junc2line[ind] = [ls1, ls2]
                    #         cnt_new += 1
            if self.verbose:
                print(f"found {cnt_new} new intercept junctions", flush=True)

    def add_junction(self, junction):
        self._junctions.append(np.array(junction))

    def add_line_seg(self, junction1, junction2):
        assert self._freeze_junction
        junc1 = np.array(junction1)
        junc2 = np.array(junction2)
        dist1, ind1 = self._kdtree.query([junc1], k=1)
        dist2, ind2 = self._kdtree.query([junc2], k=1)
        if not (dist1[0, 0] < self._eps_junc and dist2[0, 0] < self._eps_junc):
            if self.verbose:
                print(f"warn: invalid line endpoints: {junc1} -> {junc2}, ignored.")
            return
        if ind1[0, 0] == ind2[0, 0]:
            if self.verbose:
                print(f"warn: zero length line segment found ({junc1} -> {junc2}), ignored.")
            return
        self._line_segs.append(dict(
            pt1=junc1,
            pt2=junc2,
            param=fit_line([junc1, junc2])[0],
            junctions=set([ind1[0, 0], ind2[0, 0]])
        ))

    def junctions(self):
        assert self.freeze_junction and self.freeze_line_seg
        for junc in self._junctions:
            yield junc

    def line_segs(self):
        assert self.freeze_junction and self.freeze_line_seg
        for line_seg in self._line_segs:
            for ind_junc1, ind_junc2 in combinations(line_seg["junctions"], 2):
                yield self._junctions[ind_junc1], self._junctions[ind_junc2]

    def longest_line_segs(self):
        for line_seg in self._line_segs:
            yield line_seg["pt1"], line_seg["pt2"]

    @property
    def adj_mtx(self):
        mtx = np.zeros((len(self._junctions), len(self._junctions)))
        for line_seg in self._line_segs:
            for ind_junc1, ind_junc2 in combinations(line_seg["junctions"], 2):
                mtx[ind_junc1, ind_junc2] = 1
                mtx[ind_junc2, ind_junc1] = 1

        return mtx

    def line_map(self, size, scale_x=1., scale_y=1., line_width=2.):
        if isinstance(size, tuple):
            lmap = np.zeros(size, dtype=np.uint8)
        else:
            lmap = np.zeros((size, size), dtype=np.uint8)
        for line_seg in self._line_segs:
            for ind_junc1, ind_junc2 in combinations(line_seg["junctions"], 2):
                x1, y1 = self._junctions[ind_junc1]
                x2, y2 = self._junctions[ind_junc2]
                x1, x2 = int(x1 * scale_x + 0.5), int(x2 * scale_x + 0.5)
                y1, y2 = int(y1 * scale_y + 0.5), int(y2 * scale_y + 0.5)
                lmap = cv2.line(lmap, (x1, y1), (x2, y2), 255, int(line_width), cv2.LINE_AA)
        # lmap = cv2.GaussianBlur(lmap, (int(line_width), int(line_width)), 1)
        # lmap[lmap > 1] = 1
        return lmap

    @property
    def num_junctions(self):
        return len(self._junctions)

    @property
    def num_line_segs(self):
        return np.sum(
            [
                len(line_seg["junctions"])
                * (len(line_seg["junctions"]) - 1)
                / 2
                for line_seg in self._line_segs
            ]
        )


if __name__ == "__main__":
    from glob import glob
    from tqdm import trange
    data_root = "/home/ziheng/indoorDist_new"
    img = [os.path.join("train", os.path.basename(f)) for f in glob(os.path.join(data_root, "train", "*.jpg"))]
    max_junc = 0
    for item in trange(len(img)):
        lg = LineGraph().load(os.path.join(data_root, img[item][:-4] + ".lg"))
        max_junc = max(lg.num_junctions, max_junc)

    print(max_junc)