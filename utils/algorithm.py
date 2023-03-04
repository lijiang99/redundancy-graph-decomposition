import torch
import numpy as np
from scipy import fftpack
import pandas as pd
import igraph

class FilterSelection(object):
    """iteratively perform k-core decomposition, and combine the l1-norm for filter selection"""
    def __init__(self, feature_blob, weight, threshold=0.7):
        self._feature_blob = feature_blob
        self._feature_blob_num = feature_blob.shape[0]
        self._filters_num = weight.shape[0]
        # calculate the l1-norm of each filter
        l1_norms = torch.linalg.norm(weight.reshape((weight.shape[0], -1)), dim=-1, ord=1)
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
        result = {"index1": [], "index2": [], "similarity": []}
        for i in range(self._feature_blob_num):
            features = self._feature_blob[i,:,:,:]
            for j in range(self._filters_num):
                feature1 = features[j,:,:]
                for k in range(j+1, self._filters_num):
                    feature2, hamming_distance, similarity = features[k,:,:], None, None
                    hamming_distance = np.sum(self._pHash(feature1) ^ self._pHash(feature2))
                    if feature2.shape[0] < 8:
                        similarity = 1 - hamming_distance * 1.0 / (feature2.shape[0] ** 2)
                    if feature2.shape[0] >= 8:
                        similarity = 1 - hamming_distance * 1.0 / 64
                    result["index1"].append(j)
                    result["index2"].append(k)
                    result["similarity"].append(similarity)
        return pd.DataFrame(result).groupby(["index1", "index2"]).mean().reset_index()
    
    def _create_graph(self):
        """create a graph based on a specified threshold"""
        df = self.get_similarity()
        df = df[df["similarity"] >= self._threshold]
        self._graph = igraph.Graph()
        self._graph.add_vertices([i for i in range(self._filters_num)])
        self._graph.add_edges([(i, j) for i, j in zip(df["index1"], df["index2"])])
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