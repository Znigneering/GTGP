import pandas as pd,math,os
import numpy as np
from sklearn.model_selection import train_test_split

def read_tic_tac_toe(csv_path):
    csv = pd.read_csv(csv_path,header=None)
    csv = csv.replace('o',1)
    csv = csv.replace('x',-1)
    csv = csv.replace('b',0)

    csv = csv.replace('positive',1)
    csv = csv.replace('negative',0)


    return train_test_split(csv.loc[:,:8],
                            csv.loc[:,9],
                            train_size=0.8)

class Shuttle:
    def load_data(class_filter=None, binary_filter=None):
        """
            class_filter(list[int]) : filter out the rows which its class label not in list
            
            binary_filter(int) : transform class label to True/False, 
                                 only when the label equals to filter will be true
                                 
            example:
                X_train, x_test, Y_train, y_test = read_data.Shuttle.load_data(class_filter=[4,5])
                Y_train.value_counts()
                
                4    6748
                5    2458
                Name: 9, dtype: int64
                
                # shuttele
                X_train, x_test, Y_train, y_test = read_data.Shuttle.load_data(binary_filter=1)
                Y_train.value_counts()
                
                True     34108
                False     9392
                Name: 9, dtype: int64
        """
        train = pd.read_csv('data/shuttle.trn',delimiter=' ',header=None)
        test = pd.read_csv('data/shuttle.tst',delimiter=' ',header=None)
        
        train = train.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)
        
        if class_filter:
            train = train[train[9].isin(class_filter)]
            test = test[test[9].isin(class_filter)]
        if binary_filter:
            train[9] = train[9] == binary_filter
            test[9] = test[9] == binary_filter
        return train.loc[:,:8],test.loc[:,:8],train.loc[:,9],test.loc[:,9]
    
    def load_balance_data(size=2000):
        train = pd.read_csv('data/shuttle.trn',delimiter=' ',header=None)

        class_one = train[train[9] == 1][:size]

        class_two = train[train[9] == 2]

        class_two = pd.concat([class_two]*math.floor(size/len(class_two)), ignore_index=True)

        class_three = train[train[9] == 3]

        class_three = pd.concat([class_three]*math.floor(size/len(class_three)), ignore_index=True)
        
        class_four = train[train[9] == 4][:size]
        
        class_five = train[train[9] == 5][:size]

        train = pd.concat([class_one,class_two,class_three,class_four,class_five], ignore_index=True)

        train = train.sample(frac=1).reset_index(drop=True)
        
        return train.loc[:,:8],train.loc[:,9]
    
    def load_data_Exp(class_filter=None, binary_filter=None):
        """
            class_filter(list[int]) : filter out the rows which its class label not in list
            
            binary_filter(int) : transform class label to True/False, 
                                 only when the label equals to filter will be true
                                 
            example:
                X_train, x_test, Y_train, y_test = read_data.Shuttle.load_data(class_filter=[4,5])
                Y_train.value_counts()
                
                4    6748
                5    2458
                Name: 9, dtype: int64
                
                # shuttele
                X_train, x_test, Y_train, y_test = read_data.Shuttle.load_data(binary_filter=1)
                Y_train.value_counts()
                
                True     34108
                False     9392
                Name: 9, dtype: int64
        """
        train = pd.read_csv('data/shuttle.trn',delimiter=' ',header=None)
        test = pd.read_csv('data/shuttle.tst',delimiter=' ',header=None)
        
        train = train.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)
        
        if class_filter:
            train = train[train[9].isin(class_filter)]
            test = test[test[9].isin(class_filter)]
        if binary_filter:
            train[9] = train[9] == binary_filter
            test[9] = test[9] == binary_filter
        return train.loc[:,:8],test.loc[:,:8],(train.loc[:,9]-1),(test.loc[:,9]-1)
    
class CTU_13:
    def load_data(packet_num=1):
        fpath = '../data/CTU-13-Dataset/'+str(packet_num)+'/' + \
                     [fname 
                        for fname in os.listdir('../data/CTU-13-Dataset/'+str(packet_num)) 
                          if fname.find('.binetflow') > 0][0]
        
        data = pd.read_csv(fpath)
        sub_data = data[['Dur','Proto','Dir','dTos','sTos','TotPkts','TotBytes','SrcBytes','Label']]

        sub_data['Dir'] = sub_data['Dir'].replace('  <->',0)
        sub_data['Dir'] = sub_data['Dir'].replace('  <?>',1)
        sub_data['Dir'] = sub_data['Dir'].replace('   ->',2)
        sub_data['Dir'] = sub_data['Dir'].replace('   ?>',3)
        sub_data['Dir'] = sub_data['Dir'].replace('  <-',-1)
        sub_data['Dir'] = sub_data['Dir'].replace('  <?',-2)
        sub_data['Dir'] = sub_data['Dir'].replace('  who',4)

        sub_data['dTos'] = sub_data['dTos'].replace(np.nan,-1)
        sub_data['sTos'] = sub_data['sTos'].replace(np.nan,-1)

        proto = sub_data['Proto'].unique()
        for i,p in enumerate(proto):
            sub_data['Proto'] = sub_data['Proto'].replace(p,i)

        Background = sub_data[sub_data['Label'].str.contains('Background')]
        Background['Label'] = 0
        Normal = sub_data[sub_data['Label'].str.contains('Normal')]
        Normal['Label'] = 1
        Botnet = sub_data[sub_data['Label'].str.contains('Botnet')]
        Botnet['Label'] = 2

        data_with_Background = pd.concat([Background,Normal,Botnet])
        data_without_Background = pd.concat([Normal,Botnet])
        
        return data_with_Background,data_without_Background,sub_data

class Gisette:
    def load_data():
        trainx = pd.read_csv('data/gisette_train.data',delimiter=' ',header=None)[[i for i in range(5000)]]

        trainy = pd.read_csv('data/gisette_train.labels',delimiter=' ',header=None)
        trainy = trainy.replace(-1,0)
        trainy = trainy.to_numpy().transpose()[0]
        
        testx = pd.read_csv('data/gisette_valid.data',delimiter=' ',header=None)[[i for i in range(5000)]]

        testy = pd.read_csv('data/gisette_valid.labels',delimiter=' ',header=None)
        testy = testy.replace(-1,0)
        testy = testy.to_numpy().transpose()[0]
        
        testx = pd.concat([testx for i in range(6)])
        testy = np.concatenate([testy for i in range(6)])
        
        return trainx,testx,trainy,testy

class Covertype:
    def load_data():
        df = pd.read_csv('./data/covtype.data',header=None)
        x = df.loc[:,:53]

        y = df.loc[:,54]
        y = y - 1

        return train_test_split(x,y,test_size=0.1)