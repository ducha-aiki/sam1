import numpy as np
import time
import os
import math
import gc
#import theano
#import theano.tensor as T
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import paired_distances
import sys
sys.path.insert(0, '/home/old-ufo/dev/faiss')
import faiss 
from pyflann import *
#X = T.fmatrix('X')
#X_sq = T.fmatrix('X_sq')

#Y = T.fmatrix('Y')
#Y_sq = T.fmatrix('Y_sq')
#P = T.scalar('P')

#squared_euclidean_distances = X_sq.reshape((X.shape[0], 1)) + Y_sq.reshape((1, Y.shape[0])) - 2 * X.dot(Y.T)
#theano_dist_argmin = T.argmin(X_sq.reshape((X.shape[0], 1)) + Y_sq.reshape((1, Y.shape[0])) - 2 * X.dot(Y.T),axis = 1)
#gpu_euclidean = theano.function([X, Y, X_sq, Y_sq], squared_euclidean_distances)
#gpu_euc_argmin =  theano.function([X, Y, X_sq, Y_sq], theano_dist_argmin)

class KMeansClass(object):
    def __init__(self, inputdb, batch_size = 1):
        self.db = inputdb;
        self.dbsize,self.dim = inputdb.shape
        self.labels = np.zeros((self.dbsize), dtype = np.int32)
        self.dists = 9999999999999. * np.ones((self.dbsize), dtype = np.float32)
        self.norms_squared = np.zeros((self.dbsize))
        self.batch_size = batch_size;
        return
    def initializeClusterCentersRandom(self):
        idxs =  np.random.permutation(self.dbsize)
        self.centers = self.db[idxs[0:self.k],:]
        return
    def assignPointsToClustersExactGPU(self):
        centers_norms_squared = (self.centers**2).sum(axis=1).reshape(-1,1)
        if self.batch_size > 1:
            n_batches = int(math.floor(float(self.dbsize) / float(self.batch_size)))
            t = time.time()
            for batch_idx in range(n_batches):
                if batch_idx % 20 == 0:
                    print batch_idx, n_batches
                    print time.time() - t
                    t = time.time()
                curr_idxs = np.arange(batch_idx*self.batch_size,(batch_idx+1)*self.batch_size);
                query = self.db[curr_idxs,:]
                self.labels[curr_idxs] = gpu_euc_argmin(query,self.centers,
                                                        self.norms_squared[curr_idxs].reshape(-1,1), 
                                                        centers_norms_squared)
            last_batch_idxs = np.arange(n_batches*self.batch_size,self.dbsize);
            query = self.db[last_batch_idxs,:]#.reshape(-1,self.dim)
            self.labels[last_batch_idxs] = gpu_euc_argmin(query,self.centers,
                                        self.norms_squared[last_batch_idxs].reshape(-1,1), 
                                           centers_norms_squared)   
        else:
            for i in range(self.dbsize):
                query = self.db[i,:].reshape(1, -1)
                dists =  gpu_euclidean(query,self.centers)
                self.labels[i] = np.argmin(dists)
        return
    def assignPointsToClustersExact(self):
        centers_norms_squared = (self.centers**2).sum(axis=1)
        if self.batch_size > 1:
            n_batches = int(math.floor(float(self.dbsize) / float(self.batch_size)))
            t=time.time()
            for batch_idx in range(n_batches):
                if batch_idx % 20 == 0:
                    print batch_idx, n_batches
                    print time.time() - t
                    t = time.time()
                curr_idxs = np.arange(batch_idx*self.batch_size,(batch_idx+1)*self.batch_size);
                query = self.db[curr_idxs,:]
                dists =  euclidean_distances(query,self.centers, squared = True, 
                                             X_norm_squared = self.norms_squared[curr_idxs].reshape(-1,1),
                                             Y_norm_squared = centers_norms_squared)
                self.labels[curr_idxs] = np.argmin( dists, axis = 1) 
            last_batch_idxs = np.arange(n_batches*self.batch_size,self.dbsize);
            query = self.db[last_batch_idxs,:]#.reshape(-1,self.dim)
            dists =  euclidean_distances(query,self.centers, squared = True, 
                                             X_norm_squared = self.norms_squared[last_batch_idxs].reshape(-1,1),
                                             Y_norm_squared = centers_norms_squared)
            self.labels[last_batch_idxs] = np.argmin(dists, axis = 1)     
        else:
            for i in range(self.dbsize):
                query = self.db[i,:].reshape(1, -1)
                dists =  euclidean_distances(query,self.centers, squared = True, 
                                             X_norm_squared = self.norms_squared[i],
                                             Y_norm_squared = centers_norms_squared)
                self.labels[i] = np.argmin(dists)
        return
    def assignPointsToClustersFLANNKDTree(self):
        nn = FLANN()
        nn.build_index(self.centers,  algorithm='kdtree', trees=4)
        idx,dists = nn.nn_index(self.db,1)
        dists = np.array(dists)
        prev_closest_clusters = self.centers[self.labels,:]
        curr_tentative_clusters = self.centers[idx,:]
        #print prev_closest_clusters.shape
        dists2 =  paired_distances(self.db,prev_closest_clusters, metric='euclidean')
        dists3 =  paired_distances(self.db,curr_tentative_clusters, metric='euclidean')
        to_update  = np.where(dists3 < dists2)[0]
        print len(to_update), "points updated"
        #sys.exit(0)
        self.labels[to_update] = np.array(idx)[to_update]
        self.dists[to_update] = dists[to_update]
        return
    def assignPointsToClustersFLANNKMeansTree(self):
        nn = FLANN()
        nn.build_index(self.centers,  algorithm='kmeans', trees=4)
        idx,dists = nn.nn_index(self.db,1)
        dists = np.array(dists)
        prev_closest_clusters = self.centers[self.labels,:]
        curr_tentative_clusters = self.centers[idx,:]
        #print prev_closest_clusters.shape
        dists2 =  paired_distances(self.db,prev_closest_clusters, metric='euclidean')
        dists3 =  paired_distances(self.db,curr_tentative_clusters, metric='euclidean')
        to_update  = np.where(dists3 < dists2)[0]
        print len(to_update), "points updated"
        self.labels[to_update] = np.array(idx)[to_update]
        self.dists[to_update] = dists[to_update]
        #to_update  = np.where(dists < self.dists)[0]
        #self.labels[to_update] = np.array(idx)[to_update]
        #print len(to_update), "points updated"
        #self.dists[to_update] = dists[to_update]
        #self.labels = np.array(idx)
        #print self.labels.min(), self.labels.max(), self.labels.mean();
        return
    def assignPointsToClustersFAISS(self):
        nn = faiss.IndexFlatL2(self.dim)
        nn.add(self.centers)
        dists,idxs = nn.search(self.db, 1)
        print dists.shape,idxs.shape
        #idx,dists = nn.nn_index(self.db,1)
        dists = np.array(dists[:,0])
        self.labels = np.array(idxs[:,0])
        self.dists = dists
        return
    def assignPointsToClustersFAISS_GPU_IPQ(self):
        dev_no = 0
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = dev_no
        gt_index = faiss.GpuIndexFlatL2(res, self.dim, flat_config)
        coarse_quantizer = gt_index
        ncentroids = int(np.sqrt(self.k) * 4)
        index_cpu = faiss.IndexIVFPQ(coarse_quantizer, self.dim, ncentroids, 32, 8)
        # add implemented on GPU but not train
        gpuIndex =   faiss.GpuIndexIVFPQ(res, dev_no, faiss.INDICES_64_BIT,False, index_cpu)
        gpuIndex.train(self.centers)
        gpuIndex.add(self.centers)
        dists,idx = gpuIndex.search(self.db, 1)
        prev_closest_clusters = self.centers[self.labels,:]
        curr_tentative_clusters = self.centers[idx[:,0],:]
        #print prev_closest_clusters.shape
        dists2 =  paired_distances(self.db,prev_closest_clusters, metric='euclidean')
        dists3 =  paired_distances(self.db,curr_tentative_clusters, metric='euclidean')
        to_update  = np.where(dists3 < dists2)[0]
        print len(to_update), "points updated"
        self.labels[to_update] = np.array(idx[:,0])[to_update]
        return    
    def assignPointsToClustersFAISS_IPQ(self):
        ncentroids = int(np.sqrt(self.k) * 4)
        m = 8
        quantizer = faiss.IndexFlatL2(self.dim)  # this remains the same
        index = faiss.IndexIVFPQ(quantizer, self.dim, ncentroids, m, 8)
        index.train(self.centers)
        index.add(self.centers)
        dists,idx = index.search(self.db, 1)
        print idx.shape
        prev_closest_clusters = self.centers[self.labels,:]
        curr_tentative_clusters = self.centers[idx[:,0],:]
        #print prev_closest_clusters.shape
        dists2 =  paired_distances(self.db,prev_closest_clusters, metric='euclidean')
        dists3 =  paired_distances(self.db,curr_tentative_clusters, metric='euclidean')
        to_update  = np.where(dists3 < dists2)[0]
        print len(to_update), "points updated"
        self.labels[to_update] = np.array(idx[:,0])[to_update]
        #self.dists[to_update] = dists[to_update]
        return    
    def assignPointsToClustersGPUFAISS(self):
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        res = faiss.StandardGpuResources()
        nn = faiss.GpuIndexFlatL2(res, self.dim, flat_config)
        nn.add(self.centers)
        dists,idxs = nn.search(self.db, 1)
        print dists.shape,idxs.shape
        #idx,dists = nn.nn_index(self.db,1)
        dists = np.array(dists[:,0])
        self.labels = np.array(idxs[:,0])
        self.dists = dists
        return
    def updateClusterCenters(self, calcSSD):
        #centers_norms_squared = (self.centers**2).sum(axis=1)
        SSD = 0
        ###
        aa = np.sort(self.labels)
        aa_idx = np.argsort(self.labels)
        first_idx = 0
        for i in range(self.k):
            #idxs1 = self.labels == i
            idx = np.searchsorted(aa,np.int32(i+1)) 
            if (idx > 0) and (idx < len(aa) + 1 ) and (idx !=first_idx ):
                idxs = aa_idx[np.arange(first_idx,idx)]
                first_idx = idx ### This thing is faster, than self.labels == i
                if calcSSD:
                    try:
                        dists = euclidean_distances(self.centers[i,:].reshape(1,-1),self.db[idxs,:], squared = False,
                                                 Y_norm_squared = self.norms_squared[idxs]) 
                        SSD += dists.mean() / float(self.k)
                    except:
                        print i, idxs
                self.centers[i,:] = self.db[idxs,:].mean(axis = 0);
            else:
                print 'empty cluster'
                print i, idx, aa[-5:-1];
        return SSD
    def KMeans(self,k_, threshold, max_iter = 10, timings = False, method = "Exact", max_mem_size = 0, calcSSD_every_kth_iter = 5 ):
        self.k = k_;
        self.max_mem_size = max_mem_size;
        if self.max_mem_size > 0:
            self.batch_size = max(1,int(math.floor(self.max_mem_size / (self.k * self.dim))))
        else:
            self.batch_size = 1
        self.centers = np.zeros((self.k,self.dim))
        t=time.time()
        self.norms_squared = (self.db**2).sum(axis=1)
        if timings:
            print 'norms_precompute time', time.time() - t    
        print self.norms_squared.shape
        t=time.time()
        if os.path.isfile('init_centers.npy'):
            print 'loading centers pre-init'
            self.centers = np.load("init_centers.npy") 
        else:
            self.initializeClusterCentersRandom()
            np.save("init_centers.npy", self.centers)    
        SSD = 0;
        self.SSD_list = []
        if timings:
            print 'initialization time', time.time() - t
        for i in range(max_iter):
            print 'iteration ', i
            t = time.time()
            if i % calcSSD_every_kth_iter == 0:
                calcSSD = True
            else:
                calcSSD = False
            if method == "Exact":
                self.assignPointsToClustersExact()
            elif method == "ExactGPU":
                self.assignPointsToClustersExactGPU()
            elif method == "FAISSCPU":
                self.assignPointsToClustersFAISS()
            elif method == "FAISSGPU":
                self.assignPointsToClustersGPUFAISS()
            elif method == "FAISS_CPU_IPQ":
                self.assignPointsToClustersFAISS_IPQ()
            elif method == "FAISS_GPU_IPQ":
                self.assignPointsToClustersFAISS_GPU_IPQ()
            elif method == "FLANN_KDTree":
                self.assignPointsToClustersFLANNKDTree()
            elif method == "FLANN_KMeansTree":
                self.assignPointsToClustersFLANNKMeansTree()
            else: #Exact
                self.assignPointsToClustersExact()    
            if timings:
                print 'assign time', time.time() - t
                t = time.time()
            SSD_new = self.updateClusterCenters(calcSSD)
            if timings:
                print 'update centers time', time.time() - t
            self.SSD_list.append(SSD_new)
            np.save("centers_" + str(i) + ".npy", self.centers)
            np.save("labels_" + str(i) + ".npy", self.labels)
            if calcSSD:
                mean_SSD_difference =  abs((SSD_new - SSD) / (float(self.dbsize) * float(self.k)))
                print 'SSD = ', SSD_new,  'mean_SSD_difference = ', mean_SSD_difference
                SSD = SSD_new;
                gc.collect()
                if mean_SSD_difference < threshold:
                    print 'Stopping by threshold'
                    break
        return self.labels,self.centers;
    def ConstructKNNGraphForCenters(self,k_for_graph = 5):
        self.graph = np.zeros((self.k,k_for_graph), dtype = np.int32)
        aa = np.sort(self.labels)
        aa_idx = np.argsort(self.labels)
        first_idx = 0
        for label in range(self.k):
            idx = np.searchsorted(aa,np.int32(label+1)) 
            if idx > 0:
                idxs = aa_idx[np.arange(first_idx,idx)]
                first_idx = idx ### This thing is faster, than self.labels == i
                db = self.db[idxs,:]
                query = self.centers[label,:]
                dists =  euclidean_distances(query,db, squared = True)
                graph[label,:] = np.argsort(dists)[0:k_for_graph]
        return self.graph
