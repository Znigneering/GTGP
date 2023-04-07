import numpy as np
import pandas as pd
from time import time
from Node import Node

class Engine:
    def __init__(self,opset,X,Y):
        self.generation = 0
#         self.X_train,self.X_valid,self.y_train,self.y_valid = train_test_split(X,Y,train_size=0.9)
        X = X.astype('float64')
        self.opset = opset
        
        self.num_class = len(pd.unique(Y))
        self.feature_space = X.shape[1]
        
        self.vals = X.T
        self.X = X
        self.Y = Y
        self.count_label = pd.value_counts(Y).reset_index().values
        
        self.best = (-1,None)
        self.nodes = [Node(True,index=i) for i in range(self.feature_space)]
    
    def regularize(self,vals,beta):
        regularization = []
        for val in vals:
            bins_num = 0
            
            sort_label = self.Y[np.argsort(val)]
            curr_label = sort_label[0]
            for label in sort_label:
                if curr_label != label:
                    bins_num += 1
                    curr_label = label
            regularization.append(beta[0] * bins_num/self.Y.shape[0])
        return np.array(regularization)
    
    def gini_histogram(self,vals,bins,beta):
        val_by_classes = [np.transpose(np.transpose(vals)[self.Y == label]) for label,num in self.count_label]

        val_max = np.max(vals,axis=1)
        val_min = np.min(vals,axis=1)
        
        hists_by_classes = np.array([np.stack([np.histogram(val_by_classes[c][i],
                                         range=(val_min[i],val_max[i]),
                                         bins=bins)[0] / num
                             for i in range(vals.shape[0])])
                            for c,(label,num) in enumerate(self.count_label) ])
        total = np.copy(hists_by_classes[0])
        for hist in hists_by_classes[1:]:
            total += hist

        prob = [np.divide(hist,total,out = np.zeros(total.shape,dtype='float'), where=total!=0) 
                for hist in hists_by_classes]

        gini_index = np.copy(prob[0])**2
        for p in prob[1:]:
            gini_index += p**2
            
        fitness = np.sum(gini_index*total,axis=1)
        
        regularization  = self.regularize(vals,beta)
        fitness = fitness - regularization
    
        return fitness
    
    def pure_bin_histogram(self,vals,bins,beta):
        val_by_classes = [np.transpose(np.transpose(vals)[self.Y == label]) for label,num in self.count_label]

        val_max = np.max(vals,axis=1)
        val_min = np.min(vals,axis=1)
        
        hists_by_classes = np.array([np.stack([np.histogram(val_by_classes[c][i],
                                         range=(val_min[i],val_max[i]),
                                         bins=bins)[0] / num
                             for i in range(vals.shape[0])])
                            for c,(label,num) in enumerate(self.count_label) ])
        total = np.copy(hists_by_classes[0])
        for hist in hists_by_classes[1:]:
            total += hist

        prob = [np.divide(hist,total,out = np.zeros(total.shape,dtype='float'), where=total!=0) 
                for hist in hists_by_classes]

        gini_index = np.copy(prob[0])**2
        for p in prob[1:]:
            gini_index += p**2
            
        fitness = np.sum(gini_index*total,axis=1)
        
        regularization  = self.regularize(vals,beta)
        fitness = fitness - regularization
        
        
        prob_t = np.transpose(prob,axes=(1,2,0))
        pass_purity = np.any(prob_t == 0,axis=(1,2))
        fitness[pass_purity == False] = 0
        
        return fitness
    
    def validation(self,vala,valb,ya,yb,bins):        
        table,width,val_max,val_min,bins = Engine.fit_histogram(vala,ya,bins)
        meg = Engine.table_to_meg(table)
        preda = Engine.predict(vala,(meg,width,val_max,val_min,bins))
        predb = Engine.predict(valb,(meg,width,val_max,val_min,bins))

        return np.sum(preda == ya) + np.sum(predb == yb)
    
    def evolve(self,total_size,batch_size,elite_size,bins,beta,verbose):
        self.generation += 1
        
        if verbose:
            print("\tgeneration:",self.generation)
            t = time()
        
        num_batches = total_size//batch_size
        pool = self.nodes

        elites_funcs = []
        elite_sons = []
        elite_vals = []

        elites_fitness = []
        for j in range(num_batches):

            funcs = np.random.choice(list(self.opset.keys()),size=batch_size)
            arg_count = [self.opset[func] for func in funcs]
            sons = np.random.choice(pool,size = sum(arg_count))
            it = iter(sons)
            sons = [[next(it) for _ in range(arg_count[i])] for i in range(batch_size)]
            vals = [funcs[i]([self.vals[s.index] for s in sons[i]]) for i in range(batch_size)]

            vals = np.stack(vals)
            fitness = self.gini_histogram(vals,bins,beta)
    
            elites_funcs.extend(funcs)
            elite_sons.extend(sons)
            elite_vals.extend(vals)
            elites_fitness.extend(fitness)

            rank = np.argsort(elites_fitness)[::-1]

            elites_funcs = [elites_funcs[index] for index in rank[:elite_size]]
            elite_sons = [elite_sons[index] for index in rank[:elite_size]]
            elite_vals = [elite_vals[index] for index in rank[:elite_size]]
            elites_fitness = [elites_fitness[index] for index in rank[:elite_size]]

        self.update_best(self.gini_histogram(np.stack(elite_vals),bins,beta),
                         elite_vals,
                         elites_funcs,
                         elite_sons,
                         bins)
        
        # self.update_best(self.pure_bin_histogram(np.stack(elite_vals),bins,beta),
        #                  elite_vals,
        #                  elites_funcs,
        #                  elite_sons,
        #                  bins)

        for index in range(elite_size):
            node = Node(False,
                func=elites_funcs[index],
                sons=elite_sons[index],
                index=len(self.nodes),
                fit=elites_fitness[index] 
            )
            self.nodes.append(node)
            self.vals = np.append(self.vals,[elite_vals[index]],axis=0)
            # self.test_param_same(node)

        if verbose:
            print("\t",np.max(elites_fitness))
            print("\ttime",time()-t)
        return None

    def test_param_same(self,node):
        v1 = node.predict(self.X)
        v2 = self.vals[node.index]
        if np.any(v1!=v2):
            print(node.index,v1==v2)

    def update_best(self,fitness,vals,funcs,sons,bins):
        best_fitness = np.max(fitness)
        
        if best_fitness > self.best[0]:           
            index = np.argmax(fitness)
            
            node = Node(False,func=funcs[index],
                              sons=sons[index],
                              index=len(self.nodes),
                              fit=best_fitness
                       )
            
            hist = Engine.fit_histogram(vals[index],self.Y,bins)
            self.best = (best_fitness,node,hist,vals[index])
            self.nodes.append(node)
            self.vals = np.concatenate([self.vals,[vals[index]]])
            
    @staticmethod
    def fit_histogram(vals,Y,bins):
        count_label = pd.value_counts(Y).reset_index().values
        val_by_label = {label:vals[Y==label] for label,num in count_label}

        val_min = np.min(vals)
        val_max = np.max(vals)
        width = (val_max-val_min) / bins

        hists_by_classes = {label:np.histogram(val_by_label[label],
                         range=(val_min,val_max),
                         bins=bins)[0] for label,num in count_label}

        table = pd.DataFrame(hists_by_classes)

        return (table,width,val_max,val_min,bins)
    
    @staticmethod
    def table_to_meg(table):        
        return table.idxmax(axis=1)
    
    @staticmethod
    def predict(vals,hist):
        meg,width,val_max,val_min,bins = hist
        
        index = ((vals - val_min)//width).astype('int32')
        index = np.where(index >= bins,bins-1,index)
        index = np.where(index < 0,0,index)
        pred = meg[index]

        return pred.to_numpy()
    
    @staticmethod
    def fit_predict(tree,bins,X_to_fit,Y_to_fit,X_to_test):
        val_to_fit = tree.predict(X_to_fit)
        (table,width,val_max,val_min,bins) = Engine.fit_histogram(val_to_fit,Y_to_fit,bins)
        meg = Engine.table_to_meg(table)
        hist = (meg,width,val_max,val_min,bins)
        
        val_to_test = tree.predict(X_to_test)
        predictions = Engine.predict(val_to_test,hist)
        return predictions