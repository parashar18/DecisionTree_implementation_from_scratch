import pandas as pd 
import numpy as np
from numpy import log2 as log
eps = np.finfo(float).eps
import sys

def set_col_name(df):
 _,ncol=df.shape  
 val=df.keys()  
 for i in range (ncol):
     df=df.rename(columns={val[i]:i})
 return df     

def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True)
    
def find_entropy(df):
    Class = df.keys()[-1]   
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy
  
  
def find_entropy_attribute(df,attribute):
  Class = df.keys()[-1]   
  target_variables = df[Class].unique()  
  variables = df[attribute].unique()    
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2)

def find_winner(df):
    #Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
        #print(IG)
    return df.keys()[:-1][np.argmax(IG)]

def buildTree_entropy(df,tree=None): 
    Class = df.keys()[-1]   
    node = find_winner(df)
    attValue = np.unique(df[node])
    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
 
    for value in attValue:
        
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['target'],return_counts=True)                        
        
        if len(counts)==1:
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree_entropy(subtable) 
                   
    return tree

def find_varience(df):
    Class = df.keys()[-1]   
    varience = 1
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        varience *= fraction
    return varience
  
  
def find_varience_attribute(df,attribute):
  Class = df.keys()[-1]   
  target_variables = df[Class].unique()  
  variables = df[attribute].unique()    
  varience2 = 1
  for variable in variables:
      varience = 1
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den)
          varience *= fraction
      fraction2 = den/len(df)
      varience2 += fraction2*varience
  return abs(varience2)

def find_winner_varience(df):
    #Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
        IG.append(find_varience(df)-find_varience_attribute(df,key))
        #print(IG)
    return df.keys()[:-1][np.argmax(IG)]

def buildTree_varience(df,tree=None): 
    Class = df.keys()[-1]   
    node = find_winner_varience(df)
    attValue = np.unique(df[node])
    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
 
    for value in attValue:
        
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['target'],return_counts=True)                        
        
        if len(counts)==1:
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree_varience(subtable) 
                   
    return tree



def predict(inst,tree):
     
    for nodes in tree.keys():        
        
        value =inst[nodes]
        #print (value)
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break                            
    return prediction  



def find_accuracy(df):
    count=0
    nrows,_=df.shape
    #test_df_target=df.iloc[:,-1]

    for i in range(nrows):
        pre=predict(df.iloc[i,:],tree)
        #print(pre)
        #print("**")
        if pre==test_target[i]:
            count=count+1
        
    return (count/nrows)*100
#def predictotest_c300_d100[0:-2]in_accuracy():
    
    
def buildTree_entropy_preprune(df,d,depth,tree=None): 
    Class = df.keys()[-1]   
    node = find_winner(df)
    attValue = np.unique(df[node])
    #attValue=np.append(attValue,2)
    if tree is None:
        tree={}
        tree[node] = {}
           
    for value in attValue:
        #print(value)
        #if(value==2):
            #tree[node][value]=d
        #if(value!=2):
            subtable = get_subtable(df,node,value)
            #value=np.append(value,"2")
            #tree[node][2,]=d
            #print(tree[node][2,])
            clValue,counts = np.unique(subtable['target'],return_counts=True)                        
        
            if len(counts)==1:
                tree[node][value] = clValue[0]
                #tree[node][value][depth]=d                                                    
            else:        
                d+=1
                if(d==depth):
                    val=subtable.pivot_table(index="target",aggfunc='size')
                    #print(val)
                    #type(val)
                    
                    if(val[0]>=val[1]):
                        tree[node][value] = 0
                        #tree[node][value][depth]=d
                        break
                    else:
                        tree[node][value] = 1
                        #tree[node][value][depth]=d
                        break
                #print(d)
                
                tree[node][value] = buildTree_entropy_preprune(subtable,d,depth) 
                #tree[node][value][depth]=d            
    return tree

def buildTree_varience_preprune(df,d,depth,tree=None): 
    Class = df.keys()[-1]   
    node = find_winner_varience(df)
    attValue = np.unique(df[node])
    if tree is None:                    
        tree={}
        tree[node] = {}
        #tree[node][value] = {}
    for value in attValue:
        
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['target'],return_counts=True)                        
        
        if len(counts)==1:
            tree[node][value] = clValue[0]
            #tree[node][value][depth]=d
        else:        
            d+=1
            if(d==depth):
                val=subtable.pivot_table(index="target",aggfunc='size')
                #print(val)
                #type(val)
                
                if(val[0]>=val[1]):
                    tree[node][value] = 0
                    #tree[node][value][depth]=d
                    break
                else:
                    tree[node][value] = 1
                    #tree[node][value][depth]=d
                    break
            #print(d)
            #tree[node][value][depth]=d
            tree[node][value] = buildTree_varience_preprune(subtable,d,depth) 
                        
    return tree


def predict_preprune(inst,tree,depth,d):
    
    for nodes in tree.keys():        
        
        value =inst[nodes]
        #value=np.append(value,2)
        #print (value)
        
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            d+=1            
        
            if(d<depth-3):
                #d+=1
                #print(d)
                prediction = predict_preprune(inst,tree,depth,d)
            else:
                prediction = tree
                break
        else:
                prediction = tree
                break                            
    return prediction 
    
    
def find_accuracy_preprune(df,depth,tar):
    count=0
    nrows,_=df.shape
    #test_df_target=df.iloc[:,-1]

    for i in range(nrows):
        pre=predict_preprune(df.iloc[i,:],tree,depth,0)
        #print(pre)
        #print("**")
        if pre==tar[i]:
            count=count+1
        
    return (count/nrows)*100



if __name__ == '__main__':
    d=0
    tr=sys.argv[1]
    va=sys.argv[2]
    te=sys.argv[3]
    
    train=pd.read_csv(tr)
    train=set_col_name(train)
    train_target=train.iloc[:,-1]
    train.rename(columns={train.keys()[-1]:'target'}, inplace=True)
   
    valid=pd.read_csv(va)
    valid=set_col_name(valid)
    valid_target=valid.iloc[:,-1]
    valid.drop(valid.columns[-1], axis=1, inplace=True)

    test=pd.read_csv(te)
    test=set_col_name(test)
    test_target=test.iloc[:,-1]
    test.drop(test.columns[-1], axis=1, inplace=True)
    
    if(sys.argv[4]=='Entropy'): 
        if(sys.argv[5]=='0'):
            tree=buildTree_entropy(train)
            accuracy=find_accuracy(test)
            print("accuracy is ",accuracy)
        else:
            if(sys.argv[5]=='preprune'):
                l1={5,10,15,20,50,100}
                acc=[]
                for i in l1:
                    tar=valid_target
                    tree=buildTree_entropy_preprune(train,0,i)
                    #print(tree)
                    accuracy=find_accuracy_preprune(valid,i,tar)    
                    print ("Depth=",i," accuracy is= ",accuracy)
                    acc.append(accuracy)
                final_node=np.argmax(acc,axis=0)
                final_depth=acc[final_node]
                tree=buildTree_entropy_preprune(train,0,final_depth)
                tar=test_target
                final_accuracy=find_accuracy_preprune(test,final_depth,tar)    
                print("final accuracy is ",final_accuracy)
                    

    if(sys.argv[4]=='Variance'):
            if(sys.argv[5]=='0'):
                tree=buildTree_varience(train)
                accuracy=find_accuracy(test)
                print("accuracy is ",accuracy)
            else:
        
                l1={5,10,15,20,50,100}
                acc=[]
                for i in l1:
                    tar=valid_target
                    tree=buildTree_varience_preprune(train,0,i)
                    accuracy=find_accuracy_preprune(valid,i,tar)    
                    print ("Depth=",i," accuracy is= ",accuracy)
                    acc.append(accuracy)
                final_node=np.argmax(acc,axis=0)
                final_depth=acc[final_node]
                tree=buildTree_varience_preprune(train,0,final_depth)
                tar=test_target
                final_accuracy=find_accuracy_preprune(test,final_depth,tar)    
                print("final accuracy is ",final_accuracy)
    
   
         
   


