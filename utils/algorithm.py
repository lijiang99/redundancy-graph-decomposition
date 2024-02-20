import numpy as np
from scipy import fftpack
import igraph

class FilterSelection(object):
    """iteratively perform k-core decomposition, and combine the l1-norm for filter selection"""
    def __init__(self, feature_blob, weight, threshold=0.7):
        self._feature_blob = feature_blob
        self._feature_blob_num = feature_blob.shape[0]
        self._filters_num = weight.shape[0]
        # calculate the l1-norm of each filter
        l1_norms = np.linalg.norm(weight.reshape((weight.shape[0], -1)), axis=-1, ord=1)
        self._filters_l1_norms = dict(enumerate(l1_norms))
        self._threshold = threshold
        self._graph = None
    
    def _pHash(self, feature):
        """calculate the pHash of the feature map"""
        dct = fftpack.dct(fftpack.dct(feature, axis=0), axis=1)
        dct_low_freq = dct[:8,:8]
        avg = np.mean(dct_low_freq)
        return (dct_low_freq >= avg).astype(int)
    
    def get_similarity(self):
        """calculate the average similarity between the feature maps"""
        k = self._feature_blob.shape[-1] if self._feature_blob.shape[-1] < 8 else 8
        pHashs = np.zeros([self._feature_blob_num, self._filters_num, k, k]).astype(int)
        similaritys = np.zeros([self._filters_num, self._filters_num])
        # calculate the hash encoding of all feature maps
        for t in range(self._feature_blob_num):
            for i in range(self._filters_num):
                pHashs[t, i, :, :] = self._pHash(self._feature_blob[t, i, :, :])
        # calculate the average similarity for the specified sample size
        for i in range(self._filters_num):
            for j in range(i+1, self._filters_num):
                hamming_distance = 0
                for t in range(self._feature_blob_num):
                    hamming_distance += np.sum(pHashs[t, i, :, :] ^ pHashs[t, j, :, :])
                similaritys[i][j] = 1 - (hamming_distance * 1.0) / (k * k * self._feature_blob_num)
        return similaritys
    
    def _create_graph(self):
        """create a graph based on a specified threshold"""
        similaritys = self.get_similarity()
        self._graph = igraph.Graph()
        self._graph.add_vertices([i for i in range(self._filters_num)])
        self._graph.add_edges(np.argwhere(similaritys >= self._threshold))
        self._graph.vs["label"] = [i for i in range(self._filters_num)]
        return
    
    def _k_core_decompose(self):
        """k-core decomposition of the graph"""
        for k in range(self._graph.maxdegree(), 0, -1):
            sub_graph = self._graph.k_core(k)
            if sub_graph.vcount() != 0:
                return (k, sub_graph.decompose())
        return (0, self._graph)
    
    def _update_graph(self, deleted_vs):
        """remove vertices from graph"""
        vs = self._graph.vs["name"]
        deleted_ids = [vs.index(deleted_v) for deleted_v in deleted_vs]
        self._graph.delete_vertices(deleted_ids)
        return
    
    def _iter_decompose(self):
        """iterative decomposition of the graph according to k-core and l1-norm"""
        self._create_graph()
        k, sub_graphs = self._k_core_decompose()
        while k != 0:
            saved_vs, deleted_vs = [], []
            for sub_graph in sub_graphs:
                vs_map = {i: self._filters_l1_norms[i] for i in sub_graph.vs["name"]}
                # retain the vertex corresponding to the filter with the largest l1-norm
                saved_v = max(vs_map, key=vs_map.get)
                saved_vs.append(saved_v)
                vs = list(vs_map.keys())
                vs.remove(saved_v)
                deleted_vs.append(vs)
            self._update_graph(deleted_vs=[v for vs in deleted_vs for v in vs])
            k, sub_graphs = self._k_core_decompose()
        return
    
    def get_saved_filters(self):
        """get the index of the filters to retain"""
        self._iter_decompose()
        return self._graph.vs["name"]