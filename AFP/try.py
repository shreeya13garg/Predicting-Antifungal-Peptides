import pandas as pd
import numpy as np 
from sklearn import svm # import for SVM i.e. Machine Learning Model 
from sklearn.svm import LinearSVC
#from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
train_data = pd.read_csv("train.csv") 
test_data = pd.read_csv("test.csv")
mass_data  = pd.read_csv("final.csv")
mass_data_t = pd.read_csv("final1.csv")
arr=[] # train Array 
arr_t = [] # test Array


'''Feature 1 : This part deals with generating the binary profile for the entire sequence 
    it generates an array of size 20(sub_arr) corressponding to each sequence which is then stored in the final array (arr)
    this array is further passes as a feature of the SVM model later in the Code'''
e=0
for i in train_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}


    keys= amino.keys()
    for j in i:
        if j in keys:
            amino[j] +=1

    sub_arr=[]

 

    for k in amino:
        x = amino[k]
        sub_arr.append(x)

    arr.append(sub_arr)
    e+=1
#sel.fit_transform(arr)
#print(arr[0])
e=0
for i in test_data["Sequence"]:

    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}

    keys= amino.keys()
    for j in i:
        if j in keys:
            amino[j] +=1



    sub_arr=[]


    for k in amino:
        x = amino[k]
        sub_arr.append(x)


    arr_t.append(sub_arr)
    e+=1

#------------------------------------------------------------------------------------------------------------------------------------------------#
#print(arr)

Sequences=['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']
arr1=[]
arr1_t= []

for i in train_data["Sequence"]:

    pep_s= {}
    for _ in Sequences:
        for w in Sequences:
            dipep = _+w
            pep_s[dipep]=0   

    keys = pep_s.keys()     
    for j in range(0,len(i)-1 ):
        temp_pep = i[j]+i[j+1]
        if temp_pep in keys:
            pep_s[temp_pep]+=1

    sub_arr=[]
    for k in pep_s:
        x=pep_s[k]
        sub_arr.append(x)
    arr1.append(sub_arr)

#sel.fit_transform(arr1)

for i in test_data["Sequence"]:

    pep_s= {}
    for _ in Sequences:
        for w in Sequences:
            dipep = _+w
            pep_s[dipep]=0   

    keys = pep_s.keys()     
    for j in range(0,len(i)-1 ):
        temp_pep = i[j]+i[j+1]
        if temp_pep in keys:
            pep_s[temp_pep]+=1

    sub_arr=[]
    for k in pep_s:
        x=pep_s[k]
        sub_arr.append(x)
    arr1_t.append(sub_arr)
 
#-----------------------------------------------------------------------------------------------------------------------------------#
arr2=[] # train Array 
arr2_t = [] # test Array

for i in train_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}


    keys= amino.keys()
    # N5C5
    if(len(i)<10):
        for j in i:
            if j in keys:
                amino[j] +=1   

    if(len(i)>=10):
        for j in range(0,5):
            if i[j] in keys:
                amino[i[j]] +=1
    if len(i)>=10:
        for j in range(1,6):
            if i[len(i)-j] in keys:
                amino[i[len(i)-j]] +=1              

    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr2.append(sub_arr)
#sel.fit_transform(arr2)
##arr2[0])


for i in test_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}


    keys= amino.keys()

    # N5C5
    if(len(i)<10):
        for j in i:
            if j in keys:
                amino[j] +=1 

    if(len(i)>=10):
        for j in range(0,5):
            if i[j] in keys:
                amino[i[j]] +=1
    
    if(len(i)>=10):
        for j in range(1,6):
            if i[len(i)-j] in keys:
                amino[i[len(i)-j]] +=1              

    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr2_t.append(sub_arr)


#arr2_t[0])
#->---------------------------------------------------------------------------------------------------------------------------------------<-#

arr3=[]
arr3_t=[]

for i in train_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}


    keys= amino.keys()
    # N10C10

    if(len(i)<20):
        for j in i:
            if j in keys:
                amino[j] +=1 

    if(len(i)>=20):
        for j in range(0,10):
            if i[j] in keys:
                amino[i[j]] +=1
    if(len(i)>=20):
        for j in range(1,11):
            if i[len(i)-j] in keys:
                amino[i[len(i)-j]] +=1              

    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr3.append(sub_arr)
#sel.fit_transform(arr3)
#arr3[0])

for i in test_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}


    keys= amino.keys()
    # N10C10

    if(len(i)<20):
        for j in i:
            if j in keys:
                amino[j] +=1 

    if(len(i)>=20):
        for j in range(0,10):
            if i[j] in keys:
                amino[i[j]] +=1
    
    if(len(i)>=20):
        for j in range(1,11):
            if i[len(i)-j] in keys:
                amino[i[len(i)-j]] +=1              

    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr3_t.append(sub_arr)

#arr3_t[0])
#->---------------------------------------------------------------------------------------------------------------------------------------<-#

arr4=[]
arr4_t=[]


for i in train_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}


    keys= amino.keys()
    # N15C15

    if(len(i)<30):
        for j in i:
            if j in keys:
                amino[j] +=1 

    if(len(i)>=30):
        for j in range(0,15):
            if i[j] in keys:
                amino[i[j]] +=1
    if(len(i)>=30):
        for j in range(1,16):
            if i[len(i)-j] in keys:
                amino[i[len(i)-j]] +=1              

    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr4.append(sub_arr)
#sel.fit_transform(arr4)
#arr4[0])

for i in test_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}


    keys= amino.keys()
    # N15C15

    if(len(i)<30):
        for j in i:
            if j in keys:
                amino[j] +=1 

    if(len(i)>=30):
        for j in range(0,15):
            if i[j] in keys:
                amino[i[j]] +=1
    
    if(len(i)>=30):
        for j in range(1,16):
            if i[len(i)-j] in keys:
                amino[i[len(i)-j]] +=1              

    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr4_t.append(sub_arr)

#arr4_t[0])

#->---------------------------------------------------------------------------------------------------------------------------------------<-#

arr5= []
arr5_t = []

for i in train_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}

    #N5
    keys= amino.keys()

    if(len(i)<5):
        for j in i:
            if j in keys:
                amino[j] +=1 


    if(len(i)>=5):
        for j in range(0,5):
            if i[j] in keys:
                amino[i[j]] +=1 


    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr5.append(sub_arr)

#sel.fit_transform(arr5)
for i in test_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}


    keys= amino.keys()

    if(len(i)<5):
        for j in i:
            if j in keys:
                amino[j] +=1 


    if(len(i)>=5):
        for j in range(0,5):
            if i[j] in keys:
                amino[i[j]] +=1 


    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr5_t.append(sub_arr)


#->---------------------------------------------------------------------------------------------------------------------------------------<-#


arr6= []
arr6_t = []

for i in train_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
    #N10

    keys= amino.keys()

    if(len(i)<10):
        for j in i:
            if j in keys:
                amino[j] +=1 


    if(len(i)>=10):
         for j in range(0,10):
            if i[j] in keys:
                amino[i[j]] +=1 


    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr6.append(sub_arr)

#sel.fit_transform(arr6)

for i in test_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}


    keys= amino.keys()

    if(len(i)<10):
        for j in i:
            if j in keys:
                amino[j] +=1 


    if(len(i)>=10):
         for j in range(0,10):
            if i[j] in keys:
                amino[i[j]] +=1 


    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr6_t.append(sub_arr)


#->---------------------------------------------------------------------------------------------------------------------------------------<-#


arr7 = []
arr7_t =[]

for i in train_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
    #N15

    keys= amino.keys()

    if(len(i)<15):
        for j in i:
            if j in keys:
                amino[j] +=1 



    if(len(i)>=15):
         for j in range(0,15):
            if i[j] in keys:
                amino[i[j]] +=1 


    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr7.append(sub_arr)

#sel.fit_transform(arr7)
for i in test_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
    #N15

    keys= amino.keys()

    if(len(i)<15):
        for j in i:
            if j in keys:
                amino[j] +=1 

    if(len(i)>=15):
         for j in range(0,15):
            if i[j] in keys:
                amino[i[j]] +=1 


    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr7_t.append(sub_arr)


#->---------------------------------------------------------------------------------------------------------------------------------------<-#


arr8 = []
arr8_t =[]

for i in train_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
    #C5

    keys= amino.keys()

    if(len(i)<5):
        for j in i:
            if j in keys:
                amino[j] +=1 


    if(len(i)>=5):
        for j in range(1,5):
            if i[len(i)-j] in keys:
                amino[i[len(i)-j]] +=1   


    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr8.append(sub_arr)

#sel.fit_transform(arr8)
for i in test_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
    #C5

    keys= amino.keys()

    if(len(i)<5):
        for j in i:
            if j in keys:
                amino[j] +=1 


    if(len(i)>=5):
        for j in range(1,5):
            if i[len(i)-j] in keys:
                amino[i[len(i)-j]] +=1  


    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr8_t.append(sub_arr)


#->---------------------------------------------------------------------------------------------------------------------------------------<-#

arr9 = []
arr9_t =[]

for i in train_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
    #C10

    keys= amino.keys()

    if(len(i)<10):
        for j in i:
            if j in keys:
                amino[j] +=1 


    if(len(i)>=10):
        for j in range(1,10):
            if i[len(i)-j] in keys:
                amino[i[len(i)-j]] +=1    


    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr9.append(sub_arr)
#sel.fit_transform(arr9)
for i in test_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
    #C10

    keys= amino.keys()

    if(len(i)<10):
        for j in i:
            if j in keys:
                amino[j] +=1 



    if(len(i)>=10):
        for j in range(1,10):
            if i[len(i)-j] in keys:
                amino[i[len(i)-j]] +=1    


    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr9_t.append(sub_arr)

#->---------------------------------------------------------------------------------------------------------------------------------------<-#


arr10 = []
arr10_t =[]

for i in train_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
    #C15

    keys= amino.keys()

    if(len(i)<15):
        for j in i:
            if j in keys:
                amino[j] +=1 




    if(len(i)>=15):
        for j in range(1,15):
            if i[len(i)-j] in keys:
                amino[i[len(i)-j]] +=1    

    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr10.append(sub_arr)

#sel.fit_transform(arr10)
for i in test_data["Sequence"]:
    amino = {'A':0,'R':0,'N':0,'D':0,'C':0,'E':0,'Q':0,'G':0,'H':0,'I':0,'L':0,'K':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0}
    #C15

    keys= amino.keys()

    if(len(i)<15):
        for j in i:
            if j in keys:
                amino[j] +=1 

    if(len(i)>=15):
        for j in range(1,15):
            if i[len(i)-j] in keys:
                amino[i[len(i)-j]] +=1    
 


    sub_arr=[]

    for k in amino:
        x = amino[k]
        sub_arr.append(x)
    arr10_t.append(sub_arr)


#->---------------------------------------------------------------------------------------------------------------------------------------<-#


# arr11=[]
# arr11_t=[]
# for i in range(0,2550):
#     sub_arr=[]
#     val1,val2,val3=mass_data.loc[i,]
#     sub_arr.append(val1)
#     sub_arr.append(val2)
#     sub_arr.append(val3)
#     arr11.append(sub_arr)
# #sel.fit_transform(arr11)
# for i in range(0,len(mass_data_t["mass"])):
#     sub_arr=[]
#     val1,val2,val3=mass_data_t.loc[i,]
#     sub_arr.append(val1)
#     sub_arr.append(val2)
#     sub_arr.append(val3)
#     arr11_t.append(sub_arr)

# print(arr11_t)


train_data["b_profile"]=arr
test_data["b_profile"]=arr_t
train_data["dipep"]=arr1
test_data["dipep"]=arr1_t
train_data["N5C5"]=arr2
test_data["N5C5"]=arr2_t
train_data["N10C10"]=arr3
test_data["N10C10"]=arr3_t
train_data["N15C15"]=arr4
test_data["N15C15"]=arr4_t
train_data["N5"]=arr5
test_data["N5"]=arr5_t
train_data["N10"]=arr6
test_data["N10"]=arr6_t
train_data["N15"]=arr7
test_data["N15"]=arr7_t
train_data["C5"]=arr8
test_data["C5"]=arr8_t
train_data["C10"]=arr9
test_data["C10"]=arr9_t
train_data["C15"]=arr10
test_data["C15"]=arr10_t
# train_data["MAC"]=arr11
# test_data["MAC"]=arr11_t



##train_data.dtypes)
##arr1+arr)

feature_df = train_data['b_profile']+train_data['dipep']+train_data['N5C5']+train_data['N10C10']+train_data['N15C15']+train_data["N5"]+train_data["N10"]+train_data["N15"]+train_data["C5"]+train_data["C10"]+train_data["C15"]
#+train_data["MAC"]
feature_dft = test_data['b_profile']+test_data['dipep']+test_data['N5C5']+test_data['N10C10']+test_data['N15C15']+test_data["N5"]+test_data["N10"]+test_data["N15"]+test_data["C5"]+test_data["C10"]+test_data["C15"]
#+test_data["MAC"]
# #feature_df)
x_train= list(feature_df) #independent Variable
y_train= np.asarray(train_data['Lable'])# Dependent Variable
x_test = list(feature_dft)
##x_train)
#y_train)


clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1",dual=False))),
  ('classification', svm.SVC())
])


clf.fit(x_train,train_data['Lable'])
y_predict = clf.predict(x_test)
print("Predicting labels for testing data set by a trained model")
print(y_predict)
output = pd.DataFrame({'ID':test_data.ID , 'Label':y_predict})
output.to_csv('ans.csv', index = False)

print("Sub success")





