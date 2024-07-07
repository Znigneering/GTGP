import Functions
from Engine import Engine

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score,f1_score


class GTGP:

    def __init__(self,learning_rate=0.3,max_depth=3,bins=8,lam=1):
        self.learning_rate=learning_rate
        self.max_depth=max_depth
        self.bins=bins
        self.lam=lam

    def fit_one_hot(self,y):
        self.le = LabelEncoder()
        self.ohe = OneHotEncoder()
        y = self.le.fit_transform(y)
        y_one_hot = self.ohe.fit_transform(y.reshape(-1,1))
        
        return y,y_one_hot
    
    def to_one_hot(self,y):
        y = self.le.transform(y)
        y_one_hot = self.ohe.transform(y.reshape(-1,1))
        return y,y_one_hot
    
    def initial_prob(self,y_one_hot):
        self.init_p = np.sum(y_one_hot,axis=0)/y_one_hot.shape[0]
        # self.init_p = np.matrix(np.zeros(y_one_hot.shape[1]))+0.5
        
        odds = self.init_p/(1-self.init_p)

        self.init_log_odds = np.log(odds)

    def initial_first_bin(self,X):
        log_odds = np.repeat(self.init_log_odds,X.shape[0],0)
        p = np.repeat(self.init_p,X.shape[0],0)
        
        return log_odds,p
    
    def update_log_p(self,grads,log_odds,learning_rate):
        tol = 256

        # t = time()
        log_odds_1 = log_odds + learning_rate * grads
        # print("log grads",time()-t)
        
        # t = time()
        log_odds_1[log_odds_1>tol] = tol
        e = np.exp(log_odds_1)
        # print("p1",time()-t)
        
        # t = time()
        p_1 = np.divide(e,(1+e))
        # print("p1_2",time()-t)
        
        # p_1 = np.exp(log_odds_1)
        # p_1 = np.divide(p_1,(1+p_1))

        
        return log_odds_1,p_1

    def fit(self,X,y,total_size=10,elite_size = 10,epoch=1000,gp_epoch=3,tolerance=0.0001,verbose=0):
        self.num_tree = []
        self.num_nodes = []

        self.X = X
        self.y = y

        self.fit_estimators_complete = False

        self.y,self.y_one_hot = self.fit_one_hot(y)
        self.initial_prob(self.y_one_hot)
        log_odds,p = self.initial_first_bin(X)
        self.num_features = X.shape[1]
        self.stack = []
        self.losses = []

        eg = Engine(Functions.simple_opset,X,self.y,self.y_one_hot,self.learning_rate,self.bins,self.max_depth,self.lam)
        eg.initialize_nodes(log_odds,p)
        origin_features = eg.nodes.copy()
        origin_losses = eg.losses.copy()

        # self.stack.extend(eg.nodes)
        # self.losses.extend(origin_losses)

        for i in range(epoch):
            for j in range(gp_epoch):
                eg.evolve(total_size,elite_size,log_odds=log_odds,p=p,tolerance=tolerance,verbose=0)

            self.stack,self.losses = eg.adding_unique_node(self.stack,self.losses,eg.nodes[self.num_features:],tolerance=tolerance)
            for node in eg.nodes[self.num_features:]:
                node.val = None

            eg.nodes = origin_features.copy()
            eg.losses = origin_losses.copy()
            if verbose:
                self.num_tree.append(len(self.stack))
                self.num_nodes.append(sum([n.numNode for n in self.stack]))
                print(len(self.stack),sum([n.numNode for n in self.stack]),i+1)

    def fit_fixed_number(self,X,y,total_size=10,elite_size = 10,epoch=1000,gp_epoch=3,tolerance=0.01,max_nodes=1000,verbose=0):
        self.num_tree = []
        self.num_nodes = []
        self.X = X
        self.y = y

        self.fit_estimators_complete = False

        self.y,self.y_one_hot = self.fit_one_hot(y)
        self.initial_prob(self.y_one_hot)
        log_odds,p = self.initial_first_bin(X)
        self.num_features = X.shape[1]
        self.stack = []
        self.losses = []

        eg = Engine(Functions.simple_opset,X,self.y,self.y_one_hot,self.learning_rate,self.bins,self.max_depth,self.lam)
        eg.initialize_nodes(log_odds,p)
        origin_features = eg.nodes.copy()
        origin_losses = eg.losses.copy()

        i = 0
        while sum([n.numNode for n in self.stack]) <= max_nodes:
            for j in range(gp_epoch):
                eg.evolve(total_size,elite_size,log_odds=log_odds,p=p,tolerance=tolerance,verbose=0)

            self.stack,self.losses = eg.adding_unique_node(self.stack,self.losses,eg.nodes[self.num_features:],tolerance=tolerance)

            for node in eg.nodes[self.num_features:]:
                node.val = None

            eg.nodes = origin_features.copy()
            eg.losses = origin_losses.copy()
            if verbose:
                print(len(self.stack),sum([n.numNode for n in self.stack]),i+1)
                i += 1
            if i >= 1000:
                break


    def retrain_estimators(self,X_test,y_test,retrain_epoch=5,alpha=0,beta=1,gammer=0,verbose=0):
        log_odds,p = self.initial_first_bin(self.X)

        test_log_odds,test_p = self.initial_first_bin(X_test)
        y_test,y_test_one_hot = self.to_one_hot(y_test)

        sort_index = np.argsort(self.losses)
        stack = [self.stack[i] for i in sort_index]

        if verbose:
            self.train_acc = []
            self.test_acc = []

            self.train_sse = []
            self.test_sse = []

            self.train_f1_score = []
            self.test_f1_score = []
        for i in range(retrain_epoch):
            for j,tree in enumerate(stack):
                tree.estimator.lam =  self.lam + beta * tree.numNode + gammer * tree.depth
                # t = time()
                grads = tree.estimator.set_grads_bin(self.y_one_hot,p,alpha)
                # grads = tree.estimator.set_grads_bin(self.y_one_hot-p,p,alpha)
                # print("grads",time()-t)

                # t = time()
                log_odds,p = self.update_log_p(grads,log_odds,self.learning_rate)
                # print("log odd",time()-t)
                
                # t = time()
                test_grads = tree.predict_grad(X_test)
                # print("test_grads",time()-t)

                
                # t = time()
                test_log_odds,test_p = self.update_log_p(test_grads,test_log_odds,self.learning_rate)
                # print("test_log_odds",time()-t)


                self.train_p = np.array(p)
                self.test_p = np.array(test_p)
                if verbose:
                    self.train_acc.append(accuracy_score(self.y,np.argmax(self.train_p,axis=1)))
                    self.test_acc.append(accuracy_score(y_test,np.argmax(self.test_p,axis=1)))

                    self.train_sse.append(np.sum(np.power(self.y_one_hot - self.train_p,2)))
                    self.test_sse.append(np.sum(np.power(y_test_one_hot - self.test_p,2)))

                    self.train_f1_score.append(f1_score(self.y,np.argmax(self.train_p,axis=1),average='macro'))
                    self.test_f1_score.append(f1_score(y_test,np.argmax(self.test_p,axis=1),average='macro'))

                if verbose:
                    print("retrain ",i+1," tree",j,":")
                    print("\ttrain:",self.train_acc[-1],self.train_f1_score[-1],"\ttest:",self.test_acc[-1],self.test_f1_score[-1])
            
        if verbose:
            return self.train_acc,self.test_acc,self.train_sse,self.test_sse
        else:
            return 

    def print_model(self):
        print("----------------GTGP-------------")
        print("Number of Trees:",len(self.stack))
        print("Average of depth:",np.average([tree.depth for tree in self.stack]))
        print("Number of nodes:",sum([tree.numNode for tree in self.stack]))

        return len(self.stack),np.average([tree.depth for tree in self.stack]),sum([tree.numNode for tree in self.stack])

    def retrain_estimators_roc(self,X_test,y_test,retrain_epoch=20,alpha=0,beta=1,gammer=0,verbose=0):
        log_odds,p = self.initial_first_bin(self.X)

        test_log_odds,test_p = self.initial_first_bin(X_test)
        y_test,y_test_one_hot = self.to_one_hot(y_test)

        sort_index = np.argsort(self.losses)
        stack = [self.stack[i] for i in sort_index]

        if verbose:
            self.train_acc = []
            self.test_acc = []

            self.train_roc = []
            self.test_roc = []
        for i in range(retrain_epoch):
            for j,tree in enumerate(stack):
                tree.estimator.lam =  self.lam + beta * tree.numNode + gammer * tree.depth
                # t = time()
                grads = tree.estimator.set_grads_bin(self.y_one_hot,p,alpha)
                # grads = tree.estimator.set_grads_bin(self.y_one_hot-p,p,alpha)
                # print("grads",time()-t)

                # t = time()
                log_odds,p = self.update_log_p(grads,log_odds,self.learning_rate)
                # print("log odd",time()-t)
                
                # t = time()
                test_grads = tree.predict_grad(X_test)
                # print("test_grads",time()-t)
                
                # t = time()
                test_log_odds,test_p = self.update_log_p(test_grads,test_log_odds,self.learning_rate)
                # print("test_log_odds",time()-t)

                self.train_p = np.array(p)
                self.test_p = np.array(test_p)
                if verbose:
                    self.train_acc.append(accuracy_score(self.y,np.argmax(self.train_p,axis=1)))
                    self.test_acc.append(accuracy_score(y_test,np.argmax(self.test_p,axis=1)))

                    # self.train_roc.append(roc_auc_score(self.y_one_hot.toarray(),(self.train_p.T/np.sum(self.train_p,axis=1)).T))
                    # self.test_roc.append(roc_auc_score(y_test_one_hot.toarray(),(self.test_p.T/np.sum(self.test_p,axis=1)).T))
                    self.train_roc.append(roc_auc_score(self.y_one_hot.toarray(),(self.train_p)))
                    self.test_roc.append(roc_auc_score(y_test_one_hot.toarray(),(self.test_p)))

                if verbose:
                    print("retrain ",i+1," tree",j,":")
                    print("\ttrain:",self.train_acc[-1],self.train_roc[-1],"\ttest:",self.test_acc[-1],self.test_roc[-1])
            
        if verbose:
            return self.train_acc,self.test_acc,self.train_roc,self.test_roc
        else:
            return 

    # def predict_prob(self,X):
    #     test_log_odds,test_p = self.initial_first_bin(X)

    #     for node in self.stack:
    #         grads = node.predict_grad(X)
            
    #         test_log_odds,test_p = self.update_log_p(grads,test_log_odds,self.learning_rate)
    #     return test_p
    
    # def predict(self,X):
    #     test_p = self.predict_prob(X)
    #     return np.argmax(test_p,axis=1)