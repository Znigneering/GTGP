import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from time import time


class Estimator:
    def __init__(self,bins=None,lam=None,sample_weight=None):
        self.bins = bins
        self.lam = lam
        self.sample_weight = sample_weight

    def set_bins_params(self,val):
        bins = self.bins
        val_max = np.max(val)
        val_min = np.min(val)
        width = ((val_max - val_min)/bins)

        self.val_min = val_min
        self.width = width

    def fit(self,val,residual,p):
        self.set_bins_params(val)
        index = self.get_index(val)

        residual_bin = np.stack([np.sum(residual[index==i],axis=0) for i in range(self.bins)])
        cover_bin = np.stack([np.sum(np.multiply(p[index==i],1-p[index==i]),axis=0) for i in range(self.bins)])
        cover_bin = cover_bin + self.lam #lambda

        grad_bin = np.divide(residual_bin,cover_bin,where=cover_bin!=0,out=np.zeros(cover_bin.shape))
        
        self.residual_bin = residual_bin
        self.cover_bin = cover_bin
        self.grad_bin = grad_bin
        
        grads = self.get_grads(index)

        return grads
    
    def get_metrics(self,val,y):
        val = val.reshape(-1,1)
        bin_index = self.get_index(val)
        df = pd.crosstab(bin_index,y)
 
        bin_weight_by_class = (df/df.sum(axis=0))
        impurity_by_class_bin = ((df.T/df.sum(axis=1))**2)
        gains_by_class = (bin_weight_by_class.T*impurity_by_class_bin).sum(axis=1)
        
        self.metrics = gains_by_class.values
        return self.metrics

    def get_index(self,val):
        index = ((val - self.val_min)//self.width).astype('int32') if self.width !=0 else np.zeros(val.shape[0])
        index = np.where(index >= self.bins,self.bins-1,index)
        index = np.where(index < 0,0,index)

        return index

    def get_grads(self,index):
        grads = np.zeros((index.shape[0],self.grad_bin.shape[1]))
        for i in range(self.bins):
            grads[index==i] = self.grad_bin[i]
        return grads


    def predict_grad(self,val):
        index = self.get_index(val)
        grads = self.get_grads(index)
        return grads
    
class Estimator_DC:
    def __init__(self,max_depth=None,lam=None,sample_weight=None):
        self.max_depth = max_depth
        self.lam = lam
        self.sample_weight = sample_weight
        self.loss = None

    # def set_grads_bin(self,residual,p,alpha=0):
    #     # t = time()
    #     residual_bin = np.stack([np.sum(residual[self.index==i],axis=0) for i in self.terminals])
    #     sum_p = np.stack([np.sum(p[self.index==i],axis=0) for i in self.terminals])
    #     sum_y = np.stack([np.sum(y_one_hot[self.index==i],axis=0) for i in self.terminals])
    #     # print("\tresidual_bin: ",str(time()-t))
        
    #     # t = time()
    #     # cover_bin = np.stack([np.sum(np.multiply(p[self.index==i],1-p[self.index==i]),axis=0) for i in self.terminals]) + alpha*self.node_depth.reshape(-1,1)
    #     # cover_bin = np.stack([np.sum(p[self.index==i],axis=0) for i in self.terminals])
    #     # cover_bin = np.stack([np.sum(np.multiply(p[self.index==i],1-p[self.index==i]),axis=0) for i in self.terminals])
    #     cover_bin = []
    #     for i in self.terminals:
    #         a = p[self.index==i]
    #         a = np.multiply(a,1-a)
    #         c = np.sum(a,axis=0)
    #         cover_bin.append(c)
    #     cover_bin = np.stack(cover_bin)

    #     # print("\tcover_bin: ",str(time()-t))

    #     cover_bin = cover_bin + self.lam #lambda

    #     # t = time()
    #     grad_bin = np.divide(residual_bin,cover_bin,where=cover_bin!=0,out=np.zeros(cover_bin.shape))
    #     # print("\tgrad_bin: ",str(time()-t))

    #     self.residual_bin = residual_bin
    #     self.cover_bin = cover_bin
    #     self.grad_bin = grad_bin

    #     # t = time()
    #     grads = self.get_grads(self.index)
    #     # print("\tget_grads: ",str(time()-t))

    #     return grads

    def set_grads_bin(self,y_one_hot,p,alpha=0):
        g = []
        h = []

        for i in self.terminals:
            p_i = p[self.index==i]
            p_i_sum = p[self.index==i].sum(axis=0)
            y_i_sum = y_one_hot[self.index==i].sum(axis=0)

            g.append(y_i_sum - p_i_sum)
            h.append(p_i_sum - np.power(p_i,2).sum(axis=0))
        g = np.stack(g)
        h = np.stack(h)
        w = np.divide(g,h+self.lam,where=h!=0,out=np.zeros(h.shape))

        self.residual_bin = g
        self.cover_bin = h
        self.grad_bin = w

        grads = self.get_grads(self.index)
        return grads


    def fit(self,val,y):
        val = val.reshape(-1,1)

        self.clf = DecisionTreeClassifier(max_depth=self.max_depth)
        self.clf.fit(val,y,sample_weight=self.sample_weight)
        
        
        self.index = self.get_index(val)
        self.bins = self.clf.tree_.node_count
        self.terminal_bins = self.clf.get_n_leaves()

        self.terminal_index = self.clf.tree_.threshold == -2
        self.terminals = np.arange(0,self.bins,1)[self.terminal_index]

        node_depth = np.zeros(self.bins)
        for i in range(self.bins):
            index_left = self.clf.tree_.children_left[i]
            if index_left != -1:
                node_depth[index_left] = node_depth[i] + 1
            index_right = self.clf.tree_.children_right[i]
            if index_right != -1:
                node_depth[index_right] = node_depth[i] + 1
        self.node_depth = node_depth[self.terminal_index]

    def get_loss(self):
        if self.loss == None:
            impurity = self.clf.tree_.impurity[self.terminal_index]
            n_nodes_samples = self.clf.tree_.n_node_samples[self.terminal_index]
            
            self.loss = impurity@(n_nodes_samples/sum(n_nodes_samples))
        return self.loss

    def get_metrics(self,val,y):
        val = val.reshape(-1,1)
        bin_index = self.clf.apply(val)
        df = pd.crosstab(bin_index,y)
 
        bin_weight_by_class = (df/df.sum(axis=0))
        impurity_by_class_bin = ((df.T/df.sum(axis=1))**2)
        gains_by_class = (bin_weight_by_class.T*impurity_by_class_bin).sum(axis=1)
        # gains_by_class = impurity_by_class_bin.sum(axis=1)
        
        self.metrics = gains_by_class.values
        return self.metrics

    def get_index(self,val):
        val = val.reshape(-1,1)
        index = self.clf.apply(val)

        return index

    def get_grads(self,index):
        grads = np.zeros((index.shape[0],self.grad_bin.shape[1]))
        for i,terminal in enumerate(self.terminals):
            grads[index==terminal] = self.grad_bin[i]

        # grads = np.zeros((index.shape[0],self.grad_bin.shape[1]))
        # for i in range(self.bins):
        #     grads[index==i] = self.grad_bin[i]
            
        return grads


    def predict_grad(self,val):
        index = self.get_index(val)
        grads = self.get_grads(index)
        return grads