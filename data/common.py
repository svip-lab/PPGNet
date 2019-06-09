import numpy as np
from functools import lru_cache as cache


@cache(maxsize=None)
def _assert_valid_param(param):
    A, B, C = param
    assert not np.isclose(A ** 2 + B ** 2, 0), "invalid line param."
    return np.array(param) / np.sqrt(A ** 2 + B ** 2)


def assert_valid_param(param):
    return _assert_valid_param(tuple(param))


@cache(maxsize=None)
def _fit_line(pts):
    P = np.array(pts)
    P = np.hstack((P, np.ones((len(P), 1))))
    assert np.linalg.matrix_rank(P) >= 2, f"points to fit line are not valid: {P}"
    u, s, vt = np.linalg.svd(P)
    param = assert_valid_param(vt[-1])
    res = np.linalg.norm(P.dot(param)) / len(P)

    return param, res


def fit_line(pts):
    return _fit_line(tuple(tuple(pt) for pt in pts))


def dist_pts_to_line(pts, param):
    param = assert_valid_param(param)
    P = np.array([pt for pt in pts])
    P = np.hstack((P, np.ones((len(P), 1))))
    dists = np.abs(P.dot(param))

    return dists


def assert_pts_in_line(pts, param, atol=1.):
    dists = np.array(dist_pts_to_line(pts, param))
    assert np.all(dists < atol)


@cache(maxsize=None)
def _find_pt_in_line(param):
    A, B, C = assert_valid_param(param)
    if np.abs(A) > np.abs(B):
        x, y = -C / A, 0
    else:
        x, y = 0, -C / B

    return np.array([x, y])


def find_pt_in_line(param):
    return _find_pt_in_line(tuple(param))


def project_pts_on_line(pts, param):
    param = assert_valid_param(param)
    P = np.array([pt for pt in pts])
    pt0 = np.array(find_pt_in_line(param))
    e = np.array([-param[1], param[0]])
    alpha = (P - pt0).dot(e)
    P_proj = np.outer(alpha, e) + pt0

    assert P_proj.ndim == 2 and alpha.ndim == 1, f"internal error occored when project pts to line: {P_proj}, {alpha}"

    return P_proj, alpha


def find_lines_intersect(params):
    P = []
    for param in params:
        param = assert_valid_param(param)
        P.append(param)
    P = np.array(P)
    assert np.linalg.matrix_rank(P) >= 2, "lines do not intersect"
    u, s, vh = np.linalg.svd(P)
    x, y, _ = vh[-1] / vh[-1][-1]
    hpt = np.array([x, y, 1])
    dist = np.abs(P.dot(hpt)).mean()

    return np.array([x, y]), dist


@cache(maxsize=None)
def is_pt_in_line_seg(eps, pt, pt1, pt2):
    param, _ = fit_line([pt1, pt2])
    _, alphas = project_pts_on_line([pt, pt1, pt2], param)
    dist = dist_pts_to_line([pt], param)[0]
    return (dist < eps * 2) and (np.min(alphas[1:]) <= alphas[0] <= np.max(alphas[1:]))
