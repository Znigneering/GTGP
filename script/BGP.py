import numpy as np
import pandas as pd
from time import time
from Engine import Engine
from Node import Node

class BGP:
    def __init__(self,opset,X,Y):
        self.trees = []
        self.hists = []
        
        self.prev_nodes = []
        self.prev_vals = []
        self.prev_ambiguous_bin = None

        self.opset = opset
        self.fixed_X = X
        self.fixed_Y = Y
        self.X = X
        self.Y = Y
        self.stop_flag = False
        self.total_generation = 0

    def evolve(self,generation=1,total_size=100,batch_size=100,
                    elite_size=30,bins=10,beta=1,verbose=0):
        if self.stop_flag:
            return

#         E = Engine(self.opset,self.X,self.Y)
#         self.engine = E
        
        E = Engine(self.opset,self.X,self.Y)
        self.engine = E
        
        if type(self.prev_ambiguous_bin) != type(None):
            self.hists[-1][0][self.prev_ambiguous_bin] = -1
        #     E.nodes = self.prev_nodes
        #     E.vals = self.prev_vals

        # for g in range(generation):
        #     E.evolve(total_size=total_size,batch_size=batch_size,
        #                         elite_size=elite_size,bins=bins,beta=beta,verbose=verbose)

        while True:
            for g in range(generation):
                E.evolve(total_size=total_size,batch_size=batch_size,
                                    elite_size=elite_size,bins=bins,beta=beta,verbose=verbose)
            if E.best[1] != None:
                break
            else:
                E = Engine(self.opset,self.X,self.Y)
                self.engine = E
                
                # if type(self.prev_ambiguous_bin) != type(None):
                #     E.nodes = self.prev_nodes
                #     E.vals = self.prev_vals

        best = E.best[1]
        # vals = best.predict(self.X)
        # (table,width,val_max,val_min,bins) = E.fit_histogram(vals,self.Y,bins)
        vals = E.best[3]
        (table,width,val_max,val_min,bins) = E.best[2]
        
        prob = table.to_numpy()
        total = table.sum(axis=1).to_numpy()
        prob = np.divide(prob.T,total,out = np.zeros(prob.T.shape,dtype='float'), where=total!=0).T
        
        meg = Engine.table_to_meg(table)
        hist = (meg,width,val_max,val_min,bins)
        index = ((vals - val_min)//width).astype('int32')
        index = np.where(index >= bins,bins-1,index)
        index = np.where(index < 0,0,index)
        
        ambiguous_bin = np.all(prob < 1,axis=1)
        ambiguous_index = ambiguous_bin[index]
        
        self.X = self.X[ambiguous_index]
        self.Y = self.Y[ambiguous_index]
        
        E.vals = E.vals[:,ambiguous_index]

#         self.prev_nodes = E.nodes
#         self.prev_vals = E.vals
        
        self.prev_ambiguous_bin = ambiguous_bin
        self.trees.append(best)
        self.hists.append(hist)
        
        if ambiguous_bin.sum() == 0 or self.Y.shape[0] == 0:
            self.stop_flag = True
            
        if verbose:
            print('Origin Size:',self.fixed_X.shape[0],'Ambiguous Size:',self.X.shape[0],'1-A/O:',1-self.X.shape[0]/self.fixed_X.shape[0])
    
    def predict(self,X):
        cps = [np.zeros(X.shape[0])]
        ais = [np.ones(X.shape[0]).astype('bool')]

        
        for tree,hist in zip(self.trees,self.hists):
            test_vals = tree.predict(X)

            condition_pred = Engine.predict(test_vals,hist)
            
            cps.append(condition_pred)

            ambiguous_index = condition_pred == -1
            ais.append(ambiguous_index)

            X = X[ambiguous_index]

        for i in range(len(cps)-1,0,-1):
            cps[i-1][ais[i-1]] = cps[i]
        
        if False:
            return cps[0],(ais,cps)
        else:
            return cps[0]