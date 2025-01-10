import pandas as pd

times1 = pd.DataFrame(pd.read_csv("AAPL_times.csv"))
times2 = pd.DataFrame(pd.read_csv("JPM_times.csv"))
times3 = pd.DataFrame(pd.read_csv("PFE_times.csv"))
times4 = pd.DataFrame(pd.read_csv("TSLA_times.csv"))
times5 = pd.DataFrame(pd.read_csv("XOM_times.csv"))

times=list(set(times1.iloc[:,1]) & set(times2.iloc[:,1]) & set(times3.iloc[:,1]) & set(times4.iloc[:,1]) & set(times5.iloc[:,1]))
#print(pd.DataFrame(times))
#print(len(times))
l=len(times)
timestamp=[]
from datetime import time
for j in range(0,l):
    v=times[j]
    vec=v.split("T")
    t_vec=vec[1]
    time_vec=t_vec.split(":")
    hour=int(time_vec[0])
    minute=int(time_vec[1])
    second=int(time_vec[2])
    timestamp.append(time(hour,minute,second))
timestamp.sort()
#print(timestamp)
times=[]
for j in range (0,l):
    tim=timestamp[j]
    tim.strftime("%H:%M:%S")
    dat_tim=vec[0]+"T"+str(tim)
    times.append(dat_tim)
#print(pd.DataFrame(times))

# Organising the AAPL data
data1=pd.DataFrame(pd.read_csv("AAPL_OFI.csv")) # OFIs
data1_1=pd.DataFrame(pd.read_csv("AAPL_INT_OFI2.csv")) # Integrated OFIs
data1_2=pd.DataFrame(pd.read_csv("AAPL_returns.csv")) # Current returns
ncol=data1.shape[1]
data1=data1.iloc[:,1:ncol]
data1_1=data1_1.iloc[:,1:ncol]
data1_2=data1_2.iloc[:,1]
#print(data1.head())
#print(data1.iloc[:,0])
import numpy as np
dat1=np.zeros((10,l))
dat1_1=np.zeros((1,l))
dat1_2=np.zeros((l,1))
dat1_3=np.zeros((l,1))
for j in range(0,l):
    if j<=l-2:
        all_times=list(times1.iloc[:,1])
        index =  all_times.index(times[j])
        dat1[:,j]=data1.iloc[:,index]
        dat1_1[0,j]=data1_1.iloc[0,index]
        dat1_2[j,0]=data1_2.iloc[index]
        dat1_3[j,0]=data1_2.iloc[index+1]
    else:
        all_times=list(times1.iloc[:,1])
        index =  all_times.index(times[j])
        dat1[:,j]=data1.iloc[:,index]
        dat1_1[0,j]=data1_1.iloc[0,index]
        dat1_2[j,0]=data1_2.iloc[index]

#print(pd.DataFrame(dat1_1))
#print(pd.DataFrame(dat1_2))
#print(pd.DataFrame(dat1_3))
#tt=pd.DataFrame(times)
#tt.to_csv("Times.csv")


# Organising the JPM data
data2=pd.DataFrame(pd.read_csv("JPM_OFI.csv")) # OFIs
data2_1=pd.DataFrame(pd.read_csv("JPM_INT_OFI2.csv")) # Integrated OFIs
data2_2=pd.DataFrame(pd.read_csv("JPM_returns.csv")) # Current returns
ncol=data2.shape[1]
data2=data2.iloc[:,1:ncol]
data2_1=data2_1.iloc[:,1:ncol]
data2_2=data2_2.iloc[:,1]
#print(data2.head())
#print(data2.iloc[:,0])
import numpy as np
dat2=np.zeros((10,l))
dat2_1=np.zeros((1,l))
dat2_2=np.zeros((l,1))
dat2_3=np.zeros((l,1))
for j in range(0,l):
    if j<=l-2:
        all_times=list(times2.iloc[:,1])
        index =  all_times.index(times[j])
        dat2[:,j]=data2.iloc[:,index]
        dat2_1[0,j]=data2_1.iloc[0,index]
        dat2_2[j,0]=data2_2.iloc[index]
        dat2_3[j,0]=data2_2.iloc[index+1]                   
    else:
        all_times=list(times2.iloc[:,1])
        index =  all_times.index(times[j])
        dat2[:,j]=data2.iloc[:,index]
        dat2_1[0,j]=data2_1.iloc[0,index]
        dat2_2[j,0]=data2_2.iloc[index]

#print(pd.DataFrame(dat2))
#print(pd.DataFrame(dat2_1))
#print(pd.DataFrame(dat2_2))
#print(pd.DataFrame(dat2_3))
#tt=pd.DataFrame(times)
#tt.to_csv("Times.csv")


# Organising the PFE data
data3=pd.DataFrame(pd.read_csv("PFE_OFI.csv")) # OFIs
data3_1=pd.DataFrame(pd.read_csv("PFE_INT_OFI2.csv")) # Integrated OFIs
data3_2=pd.DataFrame(pd.read_csv("PFE_returns.csv")) # Current returns
ncol=data3.shape[1]
data3=data3.iloc[:,1:ncol]
data3_1=data3_1.iloc[:,1:ncol]
data3_2=data3_2.iloc[:,1]
#print(data3.head())
#print(data3.iloc[:,0])
import numpy as np
dat3=np.zeros((10,l))
dat3_1=np.zeros((1,l))
dat3_2=np.zeros((l,1))
dat3_3=np.zeros((l,1))
for j in range(0,l):
    if j<=l-2:
        all_times=list(times3.iloc[:,1])
        index =  all_times.index(times[j])
        dat3[:,j]=data3.iloc[:,index]
        dat3_1[0,j]=data3_1.iloc[0,index]
        dat3_2[j,0]=data3_2.iloc[index]
        dat3_3[j,0]=data3_2.iloc[index+1]                   
    else:
        all_times=list(times3.iloc[:,1])
        index =  all_times.index(times[j])
        dat3[:,j]=data3.iloc[:,index]
        dat3_1[0,j]=data3_1.iloc[0,index]
        dat3_2[j,0]=data3_2.iloc[index]

#print(pd.DataFrame(dat3))
#print(pd.DataFrame(dat3_1))
#print(pd.DataFrame(dat3_2))
#print(pd.DataFrame(dat3_3))
#tt=pd.DataFrame(times)
#tt.to_csv("Times.csv")


# Organising the TSLA data
data4=pd.DataFrame(pd.read_csv("TSLA_OFI.csv")) # OFIs
data4_1=pd.DataFrame(pd.read_csv("TSLA_INT_OFI2.csv")) # Integrated OFIs
data4_2=pd.DataFrame(pd.read_csv("TSLA_returns.csv")) # Current returns
ncol=data4.shape[1]
data4=data4.iloc[:,1:ncol]
data4_1=data4_1.iloc[:,1:ncol]
data4_2=data4_2.iloc[:,1]
#print(data4.iloc[:,0])
dat4=np.zeros((10,l))
dat4_1=np.zeros((1,l))
dat4_2=np.zeros((l,1))
dat4_3=np.zeros((l,1))
for j in range(0,l):
    if j<=l-2:
        all_times=list(times4.iloc[:,1])
        index =  all_times.index(times[j])
        dat4[:,j]=data4.iloc[:,index]
        dat4_1[0,j]=data4_1.iloc[0,index]
        dat4_2[j,0]=data4_2.iloc[index]
        dat4_3[j,0]=data4_2.iloc[index+1]                   
    else:
        all_times=list(times4.iloc[:,1])
        index =  all_times.index(times[j])
        dat4[:,j]=data4.iloc[:,index]
        dat4_1[0,j]=data4_1.iloc[0,index]
        dat4_2[j,0]=data4_2.iloc[index]

#print(pd.DataFrame(dat4_1))
#print(pd.DataFrame(dat4_2))
#print(pd.DataFrame(dat4_3))
#tt=pd.DataFrame(times)
#tt.to_csv("Times.csv")

# Organising the XOM data
data5=pd.DataFrame(pd.read_csv("XOM_OFI.csv")) # OFIs
data5_1=pd.DataFrame(pd.read_csv("XOM_INT_OFI2.csv")) # Integrated OFIs
data5_2=pd.DataFrame(pd.read_csv("XOM_returns.csv")) # Current returns
ncol=data5.shape[1]
data5=data5.iloc[:,1:ncol]
data5_1=data5_1.iloc[:,1:ncol]
data5_2=data5_2.iloc[:,1]
#print(data5.head())
#print(data5.iloc[:,0])
import numpy as np
dat5=np.zeros((10,l))
dat5_1=np.zeros((1,l))
dat5_2=np.zeros((l,1))
dat5_3=np.zeros((l,1))
for j in range(0,l):
    if j<=l-2:
        all_times=list(times5.iloc[:,1])
        index =  all_times.index(times[j])
        dat5[:,j]=data5.iloc[:,index]
        dat5_1[0,j]=data5_1.iloc[0,index]
        dat5_2[j,0]=data5_2.iloc[index]
        dat5_3[j,0]=data5_2.iloc[index+1]                   
    else:
        all_times=list(times5.iloc[:,1])
        index =  all_times.index(times[j])
        dat5[:,j]=data5.iloc[:,index]
        dat5_1[0,j]=data5_1.iloc[0,index]
        dat5_2[j,0]=data5_2.iloc[index]

#print(pd.DataFrame(dat5))
##print(pd.DataFrame(dat5_1))
#print(pd.DataFrame(dat5_2))
#print(pd.DataFrame(dat5_3))
#tt=pd.DataFrame(times)
#tt.to_csv("Times.csv")

# Creating the design matrix
X=np.stack((dat1[0,:],dat2[0,:],dat3[0,:],dat4[0,:],dat5[0,:]),axis=0)
X=np.transpose(X)
X=pd.DataFrame(X)
ret=np.concatenate((dat1_2,dat2_2,dat3_2,dat4_2,dat5_2),axis=1)
ret2=np.concatenate((dat1_3,dat2_3,dat3_3,dat4_3,dat5_3),axis=1)
X.columns=["AAPL","JPM","PFE","TSLA","XOM"]
ret=pd.DataFrame(ret)
ret2=pd.DataFrame(ret2)
ret.columns=["AAPL","JPM","PFE","TSLA","XOM"]
ret2.columns=["AAPL","JPM","PFE","TSLA","XOM"]

import statsmodels.api as sm

print("Best-level OFI")
#Creating the design matrix
X=sm.add_constant(X)
Self=np.zeros((5,1))
R2=np.zeros((5,1))
Cross=np.zeros((5,5))
print("************************************************************************************************************************")
print("Here are the regression models for measuring the contemporaneous cross-impact:")
print("************************************************************************************************************************")
for j in range(0,5):
    Y=ret.iloc[:,j]
    model = sm.OLS(Y, X).fit()
    Self[j]=model.params.iloc[j+1]
    R2[j]=model.rsquared_adj
    Cross[j,:]=model.params.iloc[np.array(model.params)!=Self[j]]     
    #print(model.summary())    
#print(X.head())
#print(ret.shape)
print("     ","Mean","Standard deviation",sep=" ")
print("Self:",np.mean(Self),np.std(Self),sep=" ")
print("R2:",np.mean(R2*100),np.std(R2*100),sep=" ")
print("Cross:",np.mean(Cross),np.std(Cross),sep=" ")
print("************************************************************************************************************************")
print("Here are the regression models for predictions:")
print("************************************************************************************************************************")
Self=np.zeros((5,1))
R2=np.zeros((5,1))
Cross=np.zeros((5,5))
for j in range(0,5):
    Y=ret2.iloc[:,j]
    model = sm.OLS(Y, X).fit()
    #print(model.summary())
    Self[j]=model.params.iloc[j+1]
    R2[j]=model.rsquared_adj
    Cross[j,:]=model.params.iloc[np.array(model.params)!=Self[j]]
print("     ","Mean","Standard deviation",sep=" ")
print("Self:",np.mean(Self),np.std(Self),sep=" ")
print("R2:",np.mean(R2*100),np.std(R2*100),sep=" ")
print("Cross:",np.mean(Cross),np.std(Cross),sep=" ")
print()
print()
print()


print("Integrated OFI")
# Creating the design matrix
X2=np.concatenate((dat1_1,dat2_1,dat3_1,dat4_1,dat5_1),axis=0)
X2=np.transpose(X2)
X2=pd.DataFrame(X2)
X2.columns=["AAPL","JPM","PFE","TSLA","XOM"]
X2=sm.add_constant(X2)
print("************************************************************************************************************************")
print("Here are the regression models for measuring the contemporaneous cross-impact:")
print("************************************************************************************************************************")
Self=np.zeros((5,1))
R2=np.zeros((5,1))
Cross=np.zeros((5,5))
for j in range(0,5):
    Y=ret.iloc[:,j]
    model = sm.OLS(Y, X2).fit()
    Self[j]=model.params.iloc[j+1]
    R2[j]=model.rsquared_adj
    Cross[j,:]=model.params.iloc[np.array(model.params)!=Self[j]]    
    #print(model.summary())    
print("     ","Mean","Standard deviation",sep=" ")
print("Self:",np.mean(Self),np.std(Self),sep=" ")
print("R2:",np.mean(R2*100),np.std(R2*100),sep=" ")
print("Cross:",np.mean(Cross),np.std(Cross),sep=" ")

print("************************************************************************************************************************")
print("Here are the regression models for predictions:")
print("************************************************************************************************************************")
Self=np.zeros((5,1))
R2=np.zeros((5,1))
Cross=np.zeros((5,5))
for j in range(0,5):
    Y=ret2.iloc[:,j]
    model = sm.OLS(Y, X2).fit()
    #print(model.summary())
    Self[j]=model.params.iloc[j+1]
    R2[j]=model.rsquared_adj
    Cross[j,:]=model.params.iloc[np.array(model.params)!=Self[j]]
print("     ","Mean","Standard deviation",sep=" ")
print("Self:",np.mean(Self),np.std(Self),sep=" ")
print("R2:",np.mean(R2*100),np.std(R2*100),sep=" ")
print("Cross:",np.mean(Cross),np.std(Cross),sep=" ")