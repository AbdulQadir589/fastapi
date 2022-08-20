import http
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier



app = FastAPI(title="BMI Calculator")

app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*'],
)


data=pd.read_csv('food.csv')
Breakfastdata=data['Breakfast']
BreakfastdataNumpy=Breakfastdata.to_numpy()

Lunchdata=data['Lunch']
LunchdataNumpy=Lunchdata.to_numpy()

Dinnerdata=data['Dinner']
DinnerdataNumpy=Dinnerdata.to_numpy()
Food_itemsdata=data['Food_items']


s_value=20.0
hfw=1
@app.get('/healty')
def health(age: int, veg: int, weight: float, height: int, food_type_: int):
    breakfastfoodseparated=[]
    Lunchfoodseparated=[]
    Dinnerfoodseparated=[]
        
    breakfastfoodseparatedID=[]
    LunchfoodseparatedID=[]
    DinnerfoodseparatedID=[]
        
    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i]==1:
            breakfastfoodseparated.append( Food_itemsdata[i] )
            breakfastfoodseparatedID.append(i)
        if LunchdataNumpy[i]==1:
            Lunchfoodseparated.append(Food_itemsdata[i])
            LunchfoodseparatedID.append(i)
        if DinnerdataNumpy[i]==1:
            Dinnerfoodseparated.append(Food_itemsdata[i])
            DinnerfoodseparatedID.append(i)
        
    # retrieving rows by loc method |
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val+[15]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    df=LunchfoodseparatedIDdata
    newdf=df.loc[df['Sugars'] <=s_value]
    lunchsugardata=newdf['Food_items']
    lunchdata=np.array(lunchsugardata.values.tolist())
    # retrieving rows by loc method 
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val+[15]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    df=breakfastfoodseparatedIDdata
    newdf=df.loc[df['Sugars']<=s_value]
    breakfastsugardata=newdf['Food_items']
    breakfastdata=np.array(breakfastsugardata.values.tolist())
        
    # retrieving rows by loc method 
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val+[15]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    df=DinnerfoodseparatedIDdata
    newdf=df.loc[df['Sugars'] <=s_value]
    dinnersugardata=newdf['Food_items']
    dinnerdata=np.array(dinnersugardata.values.tolist())  
    
    bmi = weight/((height/100)**2) 
    agewiseinp=0

    sugardata=np.concatenate(( dinnerdata,lunchdata,breakfastdata), axis=None)
    sugardata1=sugardata.tolist()

    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                tr=round(lp/20)  
                agecl=round(lp/20)    

        
    result = {}

    result['bmi'] = bmi
    if ( bmi < 16):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=1
    elif ( bmi >=30):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=0

    
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.to_numpy()
    ti=(bmi+agecl)/2
    
    

    Datacalorie=DinnerfoodseparatedIDdata[1:,1:len(DinnerfoodseparatedIDdata)]
    
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    XValu=np.arange(0,len(kmeans.labels_))
    # fig,axs=plt.subplots(1,1,figsize=(15,5))
    # plt.bar(XValu,kmeans.labels_)
    dnrlbl=kmeans.labels_
    # plt.title("Predicted Low-High Weigted Calorie Foods")
    
    Datacalorie=LunchfoodseparatedIDdata[1:,1:len(LunchfoodseparatedIDdata)]
    
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    #print ('## Prediction Result ##')
    #print(kmeans.labels_)
    XValu=np.arange(0,len(kmeans.labels_))
    # fig,axs=plt.subplots(1,1,figsize=(15,5))
    # plt.bar(XValu,kmeans.labels_)
    lnchlbl=kmeans.labels_
    # plt.title("Predicted Low-High Weigted Calorie Foods")
   
    Datacalorie=breakfastfoodseparatedIDdata[1:,1:len(breakfastfoodseparatedIDdata)]
    
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    XValu=np.arange(0,len(kmeans.labels_))
    # fig,axs=plt.subplots(1,1,figsize=(15,5))
    # plt.bar(XValu,kmeans.labels_)
    brklbl=kmeans.labels_
    # print (len(brklbl))
    # plt.title("Predicted Low-High Weigted Calorie Foods")
    inp=[]
    ## Reading of the Dataet
    datafin=pd.read_csv('nutrition_distriution.csv')
    datafin.head(5)
   
    dataTog=datafin.T
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    weightlosscat = dataTog.iloc[[1,2,7,8]]
    weightlosscat=weightlosscat.T
    weightgaincat= dataTog.iloc[[0,1,2,3,4,7,9,10]]
    weightgaincat=weightgaincat.T
    healthycat = dataTog.iloc[[1,2,3,4,6,7,9]]
    healthycat=healthycat.T
    weightlosscatDdata=weightlosscat.to_numpy()
    weightgaincatDdata=weightgaincat.to_numpy()
    healthycatDdata=healthycat.to_numpy()
    weightlosscat=weightlosscatDdata[1:,0:len(weightlosscatDdata)]
    weightgaincat=weightgaincatDdata[1:,0:len(weightgaincatDdata)]
    healthycat=healthycatDdata[1:,0:len(healthycatDdata)]
    
    
    weightlossfin=np.zeros((len(weightlosscat)*5,6),dtype=np.float32)
    weightgainfin=np.zeros((len(weightgaincat)*5,10),dtype=np.float32)
    healthycatfin=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    for zz in range(5):
        for jj in range(len(weightlosscat)-1):
            valloc=list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t]=np.array(valloc)
            yt.append(brklbl[jj])
            t+=1
        for jj in range(len(weightgaincat)):
            valloc=list(weightgaincat[jj])
            #print (valloc)
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r]=np.array(valloc)
            yr.append(lnchlbl[jj])
            r+=1
        for jj in range(len(healthycat)):
            valloc=list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s]=np.array(valloc)
            ys.append(dnrlbl[jj])
            s+=1

    X_test=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    
    for jj in range(len(healthycat)):
        valloc=list(healthycat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti
    
    
    X_train=healthycatfin# Features
    y_train=ys # Labels
    
    
    
    
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    
    
    X_test2=X_test
    y_pred=clf.predict(X_test)
    
    temp1=''


    for ii in range(len(y_pred)):
        if y_pred[ii]==2:
            for temp in range(len(sugardata1)):
                try:
                    if food_type_ == 0:
                        if(Food_itemsdata[ii]==sugardata1[temp] and  Food_itemsdata[ii]!= temp1):
                            temp1=Food_itemsdata[ii]

                            result[Food_itemsdata[ii]]=[Food_itemsdata[ii], data.Fats.iloc[ii]]
                        
                    elif food_type_ == 1:
                        if(Food_itemsdata[ii]==sugardata1[temp] and  Food_itemsdata[ii] != temp1):
                            result[Food_itemsdata[ii]]=[Food_itemsdata[ii], str(data.Sodium.iloc[ii])]
                    
                    else:
                        if(Food_itemsdata[ii]==sugardata1[temp] and  Food_itemsdata[ii]!= temp1):
                            temp1=Food_itemsdata[ii]

                            result[Food_itemsdata[ii]]=[Food_itemsdata[ii], data.Sugars.iloc[ii]]
                        
                except:
                    pass

    return result

@app.get('/weight-loss')
def weight_loss(age: int, veg: int, weight: float, height: int, food_type_: int):
    breakfastfoodseparated=[]
    Lunchfoodseparated=[]
    Dinnerfoodseparated=[]
        
    breakfastfoodseparatedID=[]
    LunchfoodseparatedID=[]
    DinnerfoodseparatedID=[]
        
    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i]==1:
            breakfastfoodseparated.append( Food_itemsdata[i] )
            breakfastfoodseparatedID.append(i)
        if LunchdataNumpy[i]==1:
            Lunchfoodseparated.append(Food_itemsdata[i])
            LunchfoodseparatedID.append(i)
        if DinnerdataNumpy[i]==1:
            Dinnerfoodseparated.append(Food_itemsdata[i])
            DinnerfoodseparatedID.append(i)
        
    # retrieving Lunch data rows by loc method |
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    #print(LunchfoodseparatedIDdata)
    val=list(np.arange(5,15))
    Valapnd=[0]+val+[15]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    #print(LunchfoodseparatedIDdata)
    df=LunchfoodseparatedIDdata
    newdf=df.loc[df['Sugars'] <=s_value]
    lunchsugardata=newdf['Food_items']
    lunchdata=np.array(lunchsugardata.values.tolist())
    # retrieving Breafast data rows by loc method 
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val+[15]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    df=breakfastfoodseparatedIDdata
    newdf=df.loc[df['Sugars']<=s_value]
    breakfastsugardata=newdf['Food_items']
    breakfastdata=np.array(breakfastsugardata.values.tolist())
        
    # retrieving Dinner Data rows by loc method 
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val+[15]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    df=DinnerfoodseparatedIDdata
    newdf=df.loc[df['Sugars'] <=s_value]
    dinnersugardata=newdf['Food_items']
    dinnerdata=np.array(dinnersugardata.values.tolist())

    #calculating BMI
    bmi = weight/((height/100)**2) 
    agewiseinp=0
        
    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                tr=round(lp/20)  
                agecl=round(lp/20)    

    sugardata=np.concatenate(( dinnerdata,lunchdata,breakfastdata), axis=None)
    sugardata1=sugardata.tolist()

    result = {}

    result['bmi'] = bmi
    if ( bmi < 16):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=1
    elif ( bmi >=30):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=0

    #converting into numpy array
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.to_numpy()
    ti=(clbmi+agecl)/2
    
    ## K-Means Based  Dinner Food
    Datacalorie=DinnerfoodseparatedIDdata[1:,1:len(DinnerfoodseparatedIDdata)]

    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

    XValu=np.arange(0,len(kmeans.labels_))
    
    # retrieving the labels for dinner food
    dnrlbl=kmeans.labels_

    ## K-Means Based  lunch Food
    Datacalorie=LunchfoodseparatedIDdata[1:,1:len(LunchfoodseparatedIDdata)]
    
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    XValu=np.arange(0,len(kmeans.labels_))
    
    # retrieving the labels for lunch food
    lnchlbl=kmeans.labels_
    
    ## K-Means Based  lunch Food
    Datacalorie=breakfastfoodseparatedIDdata[1:,1:len(breakfastfoodseparatedIDdata)]
    
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    XValu=np.arange(0,len(kmeans.labels_))
    
    # retrieving the labels for breakfast food
    brklbl=kmeans.labels_
    
    inp=[]
    ## Reading of the Dataet
    datafin=pd.read_csv('nutrition_distriution.csv')

    ## train set
    dataTog=datafin.T
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    weightlosscat = dataTog.iloc[[1,2,7,8]]
    weightlosscat=weightlosscat.T
    weightgaincat= dataTog.iloc[[0,1,2,3,4,7,9,10]]
    weightgaincat=weightgaincat.T
    healthycat = dataTog.iloc[[1,2,3,4,6,7,9]]
    healthycat=healthycat.T
    weightlosscatDdata=weightlosscat.to_numpy()
    weightgaincatDdata=weightgaincat.to_numpy()
    healthycatDdata=healthycat.to_numpy()
    weightlosscat=weightlosscatDdata[1:,0:len(weightlosscatDdata)]
    weightgaincat=weightgaincatDdata[1:,0:len(weightgaincatDdata)]
    healthycat=healthycatDdata[1:,0:len(healthycatDdata)]
    
    
    weightlossfin=np.zeros((len(weightlosscat)*5,6),dtype=np.float32)
    weightgainfin=np.zeros((len(weightgaincat)*5,10),dtype=np.float32)
    healthycatfin=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    for zz in range(5):
        for jj in range(len(weightlosscat)):
            valloc=list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t]=np.array(valloc)
            yt.append(brklbl[jj])
            t+=1
        for jj in range(len(weightgaincat)):
            valloc=list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r]=np.array(valloc)
            yr.append(lnchlbl[jj])
            r+=1
        for jj in range(len(healthycat)):
            valloc=list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s]=np.array(valloc)
            ys.append(dnrlbl[jj])
            s+=1

    
    X_test=np.zeros((len(weightlosscat),6),dtype=np.float32)

    
    #randomforest
    for jj in range(len(weightlosscat)):
        valloc=list(weightlosscat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti
    
    
    
    X_train=weightlossfin# Features
    y_train=yt # Labels

    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    
    #print (X_test[1])
    X_test2=X_test
    y_pred=clf.predict(X_test)
    
    temp1=''

    for ii in range(len(y_pred)):
        for temp in range(len(sugardata1)):
            try:
                if(Food_itemsdata[ii]==sugardata[temp] and  Food_itemsdata[ii]!= temp1):
                    if food_type_ == 0: 
                        if data.Fats.iloc[ii] <= 5.0:
                            temp1=Food_itemsdata[ii]
                            result[ii]=[Food_itemsdata[ii], data.Fats.iloc[ii]]
                    
                    elif food_type_==1:
                        if data.Sodium.iloc[ii] <= 80.0:
                            result[ii]=[Food_itemsdata[ii], str(data.Sodium.iloc[ii])]
                    
                    else:
                        if data.Sugars.iloc[ii] <= 8.0:
                            temp1=Food_itemsdata[ii]
                            result[ii]=[Food_itemsdata[ii], data.Sugars.iloc[ii]]
            except:
                pass

    return result


@app.get('/weight-gain')
def weight_gain(age: int, veg: int, weight: float, height: int, food_type_: int):
    breakfastfoodseparated=[]
    Lunchfoodseparated=[]
    Dinnerfoodseparated=[]
        
    breakfastfoodseparatedID=[]
    LunchfoodseparatedID=[]
    DinnerfoodseparatedID=[]
        
    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i]==1:
            breakfastfoodseparated.append( Food_itemsdata[i] )
            breakfastfoodseparatedID.append(i)
        if LunchdataNumpy[i]==1:
            Lunchfoodseparated.append(Food_itemsdata[i])
            LunchfoodseparatedID.append(i)
        if DinnerdataNumpy[i]==1:
            Dinnerfoodseparated.append(Food_itemsdata[i])
            DinnerfoodseparatedID.append(i)
        
    # retrieving rows by loc method |
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val+[15]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.T
    df=LunchfoodseparatedIDdata
    newdf=df.loc[df['Sugars'] <=s_value]
    lunchsugardata=newdf['Food_items']
    lunchdata=np.array(lunchsugardata.values.tolist())
    # retrieving rows by loc method 
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val+[15]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.T
    df=breakfastfoodseparatedIDdata
    newdf=df.loc[df['Sugars']<=s_value]
    breakfastsugardata=newdf['Food_items']
    breakfastdata=np.array(breakfastsugardata.values.tolist())  
        
    # retrieving rows by loc method 
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    val=list(np.arange(5,15))
    Valapnd=[0]+val+[15]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.T
    df=DinnerfoodseparatedIDdata
    newdf=df.loc[df['Sugars'] <=s_value]
    dinnersugardata=newdf['Food_items']
    dinnerdata=np.array(dinnersugardata.values.tolist())
    #claculating BMI
    bmi = weight/((height/100)**2)        

    sugardata=np.concatenate(( dinnerdata,lunchdata,breakfastdata), axis=None)
    sugardata1=sugardata.tolist()

    for lp in range (0,80,20):
        test_list=np.arange(lp,lp+20)
        for i in test_list: 
            if(i == age):
                tr=round(lp/20)  
                agecl=round(lp/20)
    result = {}

    result['bmi'] = bmi
    if ( bmi < 16):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=4
    elif ( bmi >= 16 and bmi < 18.5):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=3
    elif ( bmi >= 18.5 and bmi < 25):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=2
    elif ( bmi >= 25 and bmi < 30):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=1
    elif ( bmi >=30):
        result['condition'] = "Acoording to your BMI, you are Severely Underweight"
        clbmi=0


    
    DinnerfoodseparatedIDdata=DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata=LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata=breakfastfoodseparatedIDdata.to_numpy()
    ti=(bmi+agecl)/2
    
    
    ## K-Means Based  Dinner Food
    Datacalorie=DinnerfoodseparatedIDdata[1:,1:len(DinnerfoodseparatedIDdata)]
    
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    # plt.bar(XValu,kmeans.labels_)
    dnrlbl=kmeans.labels_
    # plt.title("Predicted Low-High Weigted Calorie Foods")
    
    ## K-Means Based  lunch Food
    Datacalorie=LunchfoodseparatedIDdata[1:,1:len(LunchfoodseparatedIDdata)]
    
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    XValu=np.arange(0,len(kmeans.labels_))
    # fig,axs=plt.subplots(1,1,figsize=(15,5))
    # plt.bar(XValu,kmeans.labels_)
    lnchlbl=kmeans.labels_
    # plt.title("Predicted Low-High Weigted Calorie Foods")
    
    ## K-Means Based  lunch Food
    Datacalorie=breakfastfoodseparatedIDdata[1:,1:len(breakfastfoodseparatedIDdata)]

    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    
    XValu=np.arange(0,len(kmeans.labels_))
    # fig,axs=plt.subplots(1,1,figsize=(15,5))
    # plt.bar(XValu,kmeans.labels_)
    brklbl=kmeans.labels_
    
    # plt.title("Predicted Low-High Weigted Calorie Foods")
    inp=[]
    ## Reading of the Dataet
    datafin=pd.read_csv('nutrition_distriution.csv')
    datafin.head(5)
    
    dataTog=datafin.T
    bmicls=[0,1,2,3,4]
    agecls=[0,1,2,3,4]
    weightlosscat = dataTog.iloc[[1,2,7,8]]
    weightlosscat=weightlosscat.T
    weightgaincat= dataTog.iloc[[0,1,2,3,4,7,9,10]]
    weightgaincat=weightgaincat.T
    healthycat = dataTog.iloc[[1,2,3,4,6,7,9]]
    healthycat=healthycat.T
    weightlosscatDdata=weightlosscat.to_numpy()
    weightgaincatDdata=weightgaincat.to_numpy()
    healthycatDdata=healthycat.to_numpy()
    weightlosscat=weightlosscatDdata[1:,0:len(weightlosscatDdata)]
    weightgaincat=weightgaincatDdata[1:,0:len(weightgaincatDdata)]
    healthycat=healthycatDdata[1:,0:len(healthycatDdata)]
    
   
    weightlossfin=np.zeros((len(weightlosscat)*5,6),dtype=np.float32)
    weightgainfin=np.zeros((len(weightgaincat)*5,10),dtype=np.float32)
    healthycatfin=np.zeros((len(healthycat)*5,9),dtype=np.float32)
    t=0
    r=0
    s=0
    yt=[]
    yr=[]
    ys=[]
    for zz in range(5):
        for jj in range(len(weightlosscat)):
            valloc=list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t]=np.array(valloc)
            yt.append(brklbl[jj])
            t+=1
        for jj in range(len(weightgaincat)):
            valloc=list(weightgaincat[jj])
            #print (valloc)
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r]=np.array(valloc)
            yr.append(lnchlbl[jj])
            r+=1
        for jj in range(len(healthycat)):
            valloc=list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s]=np.array(valloc)
            ys.append(dnrlbl[jj])
            s+=1

    
    X_test=np.zeros((len(weightgaincat),10),dtype=np.float32)
    # In[287]:
    for jj in range(len(weightgaincat)):
        valloc=list(weightgaincat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj]=np.array(valloc)*ti
    
    
    X_train=weightgainfin# Features
    y_train=yr # Labels
    
   
    
    
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    
   
    X_test2=X_test
    y_pred=clf.predict(X_test)
    
    
    temp1=''
    

    for ii in range(len(y_pred)):
        for temp in range(len(sugardata1)):
            try:
                if(Food_itemsdata[ii]==sugardata[temp] and  Food_itemsdata[ii]!= temp1):
                    temp1=Food_itemsdata[ii]
                    
                    if food_type_ == 0:
                        if data.Fats.iloc[ii] > 5.0:
                            result[ii]=[Food_itemsdata[ii], data.Fats.iloc[ii]]

                    if food_type_ == 1:
                        
                        if data.Sodium.iloc[ii] > 80.0:
                            result[ii]=[Food_itemsdata[ii], str(data.Sodium.iloc[ii])]
                    
                    if food_type_ == 2:
                        if data.Sugars.iloc[ii] > 6.0:
                            result[ii]=[Food_itemsdata[ii], data.Sugars.iloc[ii]]
            except:
                pass

        if int(veg)==1:
            datanv=['Chicken Burger']
    return result
