import numpy as np
import pandas as pd
from time import time
from Node import Node
from Estimator import Estimator_DC,Estimator

class Engine:
    def __init__(self,opset,X,Y,Y_one_hot,learning_rate=0.3,bins=3,max_depth=3,lam=1):
        self.generation = 0
        
        X = X.astype('float64')
        self.opset = opset
        
        self.feature_space = X.shape[1]
        
        self.vals = X.T
        self.X = X
        self.Y = Y
        self.Y_one_hot = Y_one_hot

        base_gini = (pd.value_counts(Y)/Y.shape)
        self.base_gini = sum(base_gini*(1-base_gini))

        self.learning_rate = learning_rate

        self.bins = bins
        self.max_depth = max_depth
        self.lam = lam

    def initialize_nodes(self,log_odds,p):
        self.losses = []
        self.nodes = []

        for i in range(self.feature_space):
            val = self.X[:,i]
            loss = -1

            node = Node(terminal=True,val=val,index=i,fit=-1,estimator=None)
            self.nodes.append(node)
            self.losses.append(loss)

        self.nodes,self.losses = self.sort_stack_losses(self.nodes,self.losses)

    def sort_stack_losses(self,nodes,losses):
        index = np.argsort(losses)
        nodes = [nodes[i] for i in index]
        losses = [losses[i] for i in index]

        return nodes,losses
    
    def adding_unique_node(self,stack,losses,new_nodes,tolerance=1e-5):
        for new_node in new_nodes:
            has_added_node = False
            no_similar_node = True

            for i in range(len(stack)-1,-1,-1):
                node = stack[i]
                if self.is_similar_metrics(new_node,node,tolerance):
                    no_similar_node = False
                    if new_node.numNode < node.numNode:
                        stack.pop(i)
                        losses.pop(i)

                        if not has_added_node:
                            stack.append(new_node)
                            losses.append(new_node.estimator.loss)
                            has_added_node = True
                    
                    # if new_node.estimator.get_loss() < node.estimator.get_loss():
                    #     stack.pop(i)
                    #     losses.pop(i)

                    #     if not has_added_node:
                    #         stack.append(new_node)
                    #         losses.append(new_node.estimator.loss)
                    #         has_added_node = True

            if no_similar_node and not has_added_node:
                stack.append(new_node)
                losses.append(new_node.estimator.loss)

        return stack,losses
    
    def is_similar_metrics(self,node_a,node_b,tolerance):
        metrics_a = node_a.estimator.metrics
        metrics_b = node_b.estimator.metrics

        count = 0
        for i in range(len(metrics_a)):
            if abs(metrics_a[i]-metrics_b[i]) < tolerance:
                count += 1

        return count == len(metrics_a)


    def calculate_fitness(self,vals,sample_weight):
        fitness = []
        estimators = []
        for val in vals:
            est = Estimator_DC(self.max_depth,self.lam,sample_weight)
            est.fit(val,self.Y)
            
            # est = Estimator(self.bins,self.lam)
            # est.fit(val,self.Y,self.p)

            loss = est.get_loss()
            
            fitness.append(loss)
            estimators.append(est)

        return fitness,estimators

    def evolve(self,total_size,elite_size,log_odds,p,tolerance=1e-3,verbose=0):
        self.generation += 1
        # sample_weight = np.ravel(np.sum(np.power(p-self.Y_one_hot,2),axis=1))
        sample_weight = None
        
        if verbose:
            print("\tgeneration:",self.generation)
            t = time()
        
        pool = self.nodes

        funcs = np.random.choice(list(self.opset.keys()),size=total_size)
        arg_count = [self.opset[func] for func in funcs]
        sons = np.random.choice(pool,size = sum(arg_count))
        it = iter(sons)
        sons = [[next(it) for _ in range(arg_count[i])] for i in range(total_size)]
        vals = [funcs[i]([s.val for s in sons[i]]) for i in range(total_size)]

        vals = np.stack(vals)
        fitness,estimators = self.calculate_fitness(vals,sample_weight)
        
        rank = np.argsort(fitness)

        new_nodes = [Node(False,
                            func=funcs[index],
                            sons=sons[index],
                            val=vals[index],
                            fit=fitness[index],
                            estimator=estimators[index]
                        ) for index in rank[:elite_size]]
        for node in new_nodes:
            node.estimator.get_metrics(node.val,self.Y)
        
        # nodes,losses = self.adding_unique_node(stack=self.nodes[self.feature_space:],losses=self.losses[self.feature_space:],new_nodes=new_nodes,tolerance=tolerance)
        # self.nodes += new_nodes
        # self.losses += [node.estimator.loss for node in new_nodes]

        self.nodes += new_nodes
        self.losses += [node.estimator.loss for node in new_nodes]

        if verbose:
            print(len(self.nodes),"\ttime",time()-t)
        return None