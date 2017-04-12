import numpy as np
import time
import os
import math
import cPickle as pickle
import gc
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import paired_distances
import sys
sys.path.insert(0, '/home/old-ufo/dev/faiss')
import faiss 
from pyflann import *
sys.path.insert(0, '/home/ubuntu/dev/opencv-3.1/build/lib')
import cv2
import matplotlib.pyplot as plt
def readSmallDescAndFnames(fname, txtfname):
    X = np.fromfile(fname, dtype=np.float32, count=-1, sep='').reshape((-1,128))
    with open(txtfname) as ff:
        lines = ff.readlines()
    return X,lines

def BuildKNNGraphByFLANNKDTree(db, k):
    dbsize, dim = db.shape
    nn = FLANN()
    nn.build_index(db,  algorithm='kdtree', trees=4)
    idx,dists = nn.nn_index(db,k + 1) # first is object itself
    dists2 = np.zeros(dists.shape)
    for i in range(dbsize):
        curr_tentative_clusters = db[idx[i,:],:]
        dists2[i,:] =  euclidean_distances(db[i,:].reshape(1, -1),curr_tentative_clusters)
    return idx[:,1:],dists2[:,1:]
def BuildKNNGraphByFLANNKMeansTree(db,k):
    dbsize, dim = db.shape
    nn = FLANN()
    nn.build_index(db,  algorithm='kmeans', trees=4)
    idx,dists = nn.nn_index(db,k + 1) # first is object itself
    dists2 = np.zeros(dists.shape)
    for i in range(dbsize):
        curr_tentative_clusters = db[idx[i,:],:]
        dists2[i,:] =  euclidean_distances(db[i,:].reshape(1, -1),curr_tentative_clusters)
    return idx[:,1:],dists2[:,1:]
def BuildKNNGraphByFAISS_CPU_IPQ(db,k):
    dbsize, dim = db.shape
    m = 8
    ncentroids = int(np.sqrt(dbsize) * 4)
    quantizer = faiss.IndexFlatL2(dim)  # this remains the same
    index = faiss.IndexIVFPQ(quantizer, dim, ncentroids, m, 8) 
    index.train(db)
    index.add(db)    
    dists,idx = index.search(db, k+1)
    dists = np.nan_to_num(dists)
    dists = np.clip(dists,0, 100)
    dists2 = np.zeros(dists.shape)
    for i in range(dbsize):
        curr_tentative_clusters = db[idx[i,:],:]
        dists2[i,:] =  euclidean_distances(db[i,:].reshape(1, -1),curr_tentative_clusters)
    return idx[:,1:],dists2[:,1:]
def BuildKNNGraphByFAISS_GPU(db,k):
    dbsize, dim = db.shape
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    res = faiss.StandardGpuResources()
    nn = faiss.GpuIndexFlatL2(res, dim, flat_config)
    nn.add(db)
    dists,idx = nn.search(db, k+1)
    return idx[:,1:],dists[:,1:]
def get_full_path(q_idx, imglist):
    return "data/oxc-complete/" + imglist[q_idx].strip() + '.jpg'
def visualizeKNNImages(knngraph_dict, q_idx = 0, methods = ["PQ", "Exact"], thumb_size = 100, k = 5, pad = 5, imglist = []):
    output_img = np.zeros(( (thumb_size + 2*pad) * len(methods) ,(thumb_size + 2*pad) * (k+1) ,3)); 
    for m_idx in range(len(methods)):
        m = methods[m_idx]
        print get_full_path(q_idx, imglist)
        q = cv2.imread(get_full_path(q_idx, imglist))
        q = cv2.resize(q, (thumb_size, thumb_size))
        output_img[(pad + 1)*m_idx  + thumb_size*m_idx : (pad + 1)*m_idx  + thumb_size*(m_idx+1),
                  pad: pad + thumb_size,:] = q
        for nn_idx in range(k):
            img = cv2.imread(get_full_path(knngraph_dict[m][0][q_idx, nn_idx], imglist))
            img = cv2.resize(img, (thumb_size, thumb_size))
            output_img[(pad + 1)*m_idx  + thumb_size*m_idx : (pad + 1)*m_idx  + thumb_size*(m_idx+1),
                  (2*pad + thumb_size)*(nn_idx + 1):(2*pad + thumb_size)*(nn_idx + 1) + thumb_size,:] = img
            
    return output_img

db_SIFT, db_fnames = readSmallDescAndFnames('data/imgdesc105k.dat', 'data/imagenames105k.txt')
db_CNN, db_fnames = readSmallDescAndFnames('data/CNNdesc105k.dat', 'data/imagenames105k.txt')

knn_graph_dict = {"SIFT": {}, "CNN": {}}
k = 20
t = time.time()
knn_graph_dict["CNN"]["KD-Tree"] = BuildKNNGraphByFLANNKDTree(db_CNN, k)
print time.time() - t
t = time.time()
knn_graph_dict["CNN"]["KMeans-Tree"] = BuildKNNGraphByFLANNKMeansTree(db_CNN, k)
print time.time() - t
t = time.time()
knn_graph_dict["CNN"]["Exact"] = BuildKNNGraphByFAISS_GPU(db_CNN, k)
print time.time() - t
t = time.time()
t = time.time()
knn_graph_dict["SIFT"]["KD-Tree"] = BuildKNNGraphByFLANNKDTree(db_SIFT, k)
print time.time() - t
t = time.time()
knn_graph_dict["SIFT"]["KMeans-Tree"] = BuildKNNGraphByFLANNKMeansTree(db_SIFT, k)
print time.time() - t
t = time.time()
knn_graph_dict["SIFT"]["Exact"] = BuildKNNGraphByFAISS_GPU(db_SIFT, k)
print time.time() - t
t = time.time()
t = time.time()
knn_graph_dict["CNN"]["PQ"] = BuildKNNGraphByFAISS_CPU_IPQ(db_CNN, k)
print time.time() - t

t = time.time()
knn_graph_dict["SIFT"]["PQ"] = BuildKNNGraphByFAISS_CPU_IPQ(db_SIFT, k)
print time.time() - t
pickle.dump(knn_graph_dict, open("KNNgraph2.pickle", 'wb'), protocol = 2)
methods = ["Exact", "KD-Tree", "KMeans-Tree","PQ"]
dd = "CNN"
for i in range(5):
    vis1 = visualizeKNNImages(knn_graph_dict[dd],q_idx = 1000*i, methods = methods
                              , thumb_size = 100, k = 10, pad = 5, imglist = db_fnames)
    cv2.imwrite("compare_" + dd + "_" + str(i) + ".png",vis1)


plt.figure()
x = range(1,k+1)
dd = "CNN"
plt.plot(x, np.mean(knn_graph_dict[dd]["Exact"][1], axis = 0), 'x',
        x, np.mean(knn_graph_dict[dd]["KD-Tree"][1], axis = 0),'x',
        x, np.mean(knn_graph_dict[dd]["KMeans-Tree"][1], axis = 0),'x',
        x, np.mean(knn_graph_dict[dd]["PQ"][1], axis = 0),'x',
        )
plt.legend(methods,prop={'size':12},loc = 'best')
plt.xlabel('NN number')
plt.ylabel('Average L2 distance')
plt.savefig(dd + "_avg_knn_dist.eps")


plt.figure()
x = range(1,k+1)
dd = "CNN"
plt.plot(x, np.mean(knn_graph_dict[dd]["Exact"][1], axis = 0), 'x',
        x, np.mean(knn_graph_dict[dd]["KD-Tree"][1], axis = 0),'x',
        x, np.mean(knn_graph_dict[dd]["KMeans-Tree"][1], axis = 0),'x',
        x, np.mean(knn_graph_dict[dd]["PQ"][1], axis = 0),'x',
        )
plt.legend(methods,prop={'size':12},loc = 'best')
plt.xlabel('NN number')
plt.ylabel('Average L2 distance')
plt.savefig(dd + "_avg_knn_dist.eps")
#print np.mean(knn_graph_dict[dd]["Exact"][1], axis = 0)
