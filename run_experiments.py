#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
import numpy as np
from KM import KMeansClass
import cPickle as pickle
def readSmallDescAndFnames():
    X = np.fromfile('imagedesc.dat', dtype=np.float32, count=-1, sep='').reshape((-1,128))
    with open('imagenames.txt') as ff:
        lines = ff.readlines()
    return X,lines
def readSIFT2M():
    return np.fromfile('SIFT.dat', dtype=np.uint8, count=-1, sep='').reshape((-1,128))
small_db,Names = readSmallDescAndFnames()
SIFT2M = readSIFT2M().astype(np.float32)
k = 32000
th = 0#0.00000001
#"ExactGPU", 
methods = ["FLANN_KDTree","FLANN_KMeansTree", "FAISS_GPU_IPQ", "FAISS_CPU_IPQ", "FAISSGPU", "ExactGPU", "FAISSCPU",  "Exact"]
KMO = KMeansClass(SIFT2M)
results = dict()
calcSSD_every_kth_iter = 1;
for method in methods:
    print method
    current_results = {}
    lab,cent = KMO.KMeans(k,th,max_iter = 30, timings = True,
                               method = method, max_mem_size = 200 * 128 * 1000 * 1000, calcSSD_every_kth_iter = calcSSD_every_kth_iter)
    current_results["labels"] = lab
    current_results["centers"] = cent
    current_results["SSD"] = KMO.SSD_list
    results[method] = current_results
    pickle.dump(results, open("results_new_fix_faiss_ipq.pickle", "wb"), protocol = 2)
