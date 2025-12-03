import numpy as np
import cv2
import networkx as nx
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from skimage.color import rgb2lab
from skimage.segmentation import slic


def compute_spectral_gap(G):
    A = nx.to_scipy_sparse_matrix(G, weight='weight', format='csr')
    L = csgraph.laplacian(A, normed=False)
    vals, _ = eigsh(L, k=3, which='SM')
    vals = np.sort(vals)
    return float(vals[1]), vals


def iou_pixel(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return inter / union if union > 0 else 0.0


def iou_superpixel(sp_labels, pred_nodes, gt_nodes):
    sp_pred = pred_nodes[sp_labels]
    sp_gt = gt_nodes[sp_labels]
    inter = np.logical_and(sp_pred, sp_gt).sum()
    union = np.logical_or(sp_pred, sp_gt).sum()
    return inter / union if union > 0 else 0.0


def build_superpixel_graph(img_rgb, n_segments=200, compactness=10,
                           beta=0.05, lambda_spatial=1.0):
    sp = slic(img_rgb, n_segments=n_segments, compactness=compactness, start_label=0)
    h, w = sp.shape
    n_nodes = sp.max() + 1

    img_lab = rgb2lab(img_rgb)
    node_color = np.zeros((n_nodes, 3))
    node_centroid = np.zeros((n_nodes, 2))
    counts = np.zeros(n_nodes)

    for k in range(n_nodes):
        mk = (sp == k)
        if mk.sum() == 0:
            continue
        node_color[k] = img_lab[mk].mean(0)
        ys, xs = np.where(mk)
        node_centroid[k] = [ys.mean(), xs.mean()]
        counts[k] = mk.sum()

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    for y in range(h - 1):
        for x in range(w - 1):
            a = sp[y, x]
            b1 = sp[y, x + 1]
            b2 = sp[y + 1, x]
            if a != b1: G.add_edge(a, b1)
            if a != b2: G.add_edge(a, b2)

    for u, v in G.edges():
        dc = np.linalg.norm(node_color[u] - node_color[v]) ** 2
        ds = np.linalg.norm(node_centroid[u] - node_centroid[v]) ** 2
        ds_norm = ds / (h * h + w * w)
        wgt = np.exp(-beta * dc) * np.exp(-lambda_spatial * ds_norm)
        G[u][v]['weight'] = float(wgt)

    centroids_norm = node_centroid.copy()
    centroids_norm[:, 0] /= h
    centroids_norm[:, 1] /= w

    return G, sp, node_color, centroids_norm, counts
