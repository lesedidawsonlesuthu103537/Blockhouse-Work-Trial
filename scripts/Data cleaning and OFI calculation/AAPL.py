import csv
# Reading the data in
with open("AAPL_raw_data.csv","r") as csvfile:
    csv_reader=csv.reader(csvfile)
    data=[]
    for row in csv_reader:
        data.append(row)

import pandas as pd
data=pd.DataFrame(data)

# Adjusting dataset to remove the first row and label the columns
nr=data.shape[0]
ncol=data.shape[1]
newdata=data.iloc[1:nr,1:ncol]
newdata.columns=data.iloc[0,1:ncol]
data=newdata
print(data.columns)
print(data.head())

#Creating minute labels for each data point to group them by the minute and to use in calculation of 1 minute OFIs
rows=data.shape[0]
dates=[]
for j in range(0,rows):
    v=data.iloc[j,0]
    v=v.split(".")
    v=v[0]
    pq=v.split(":")
    pq[2]="00"
    pq=":".join(pq)
    dates.append(pq)
print(dates[0:5])

# Finding the unique elements in the dates matrix
import numpy as np
j=np.array(dates)
uniq=np.unique(j)
print(uniq[0:5])


# OFI Calculation
tot_obs=len(uniq)
ofi_mat=np.zeros((10,tot_obs))
price=[]
k=data.iloc[np.array(dates)==uniq[0],:]
print(k)
for jjk in range(0,tot_obs):
    dat=data.iloc[np.array(dates)==uniq[jjk],:]
    l=dat.columns
    m=len(l)
    g=[]
    g.append(l[0])
    for w in range(12,m):
        g.append(l[w])
    g=list(g)
    fin_col=[]
    for name in g:
        v=name.split("_")
        if len(v)>1:
            if v[1]!='ct':
                fin_col.append(name)
    dat1=dat[fin_col]
    dat1=dat1[(dat1 != '').all(1)] # Removing rows with nan
    m=dat1.shape[0]
    n=dat1.shape[1]
    import numpy as np
    OFI_bid=np.zeros((m,10))
    
    # Calculating the mid price and storing it in a vector
    price.append((dat1.iat[0,1]+dat1[0,2])/2)
    if jjk==tot_obs-1:
        price.append((dat1.iat[m-1,1]+dat1[m-1,2])/2)
    from numpy import nan

    # Calculation of bid and ask order flows as specified in the report
    for i in range(1,m):
        for j in range (1,n,4):#n-2
            p=int((j+3)/4-1)
            if eval(str(dat1.iat[i,j]))>eval(str(dat1.iat[i-1,j])):
                OFI_bid[i-1,p]=int(dat1.iat[i,j+2])
            elif eval(str(dat1.iat[i,j]))==eval(str(dat1.iat[i-1,j])):
                OFI_bid[i-1,p]=int(dat1.iat[i,j+2])-int(dat1.iat[i-1,j+2])
            else:
                OFI_bid[i-1,p]=-1*int(dat1.iat[i,j+2])
    
    import numpy as np
    OFI_ask=np.zeros((m,10)) 
    from numpy import nan
    for i in range(1,m):
        for j in range (2,n+1,4):
            p=int((j+2)/4-1)
            if eval(str(dat1.iat[i,j]))>eval(str(dat1.iat[i-1,j])):
                OFI_ask[i-1,p]=-1*int(dat1.iat[i,j+2])
            elif eval(str(dat1.iat[i,j]))==eval(str(dat1.iat[i-1,j])):
                OFI_ask[i-1,p]=int(dat1.iat[i,j+2])-int(dat1.iat[i-1,j+2])
            else:
                OFI_ask[i-1,p]=int(dat1.iat[i,j+2])
    import numpy as np
    # Calculation of best-level and deeper-level OFIs
    OFI=np.sum(OFI_bid-OFI_ask, axis=0)

        
    # Calculating Q
    import numpy as np
    #print("Here is the raw OFI's for the ask prices")
    Qmat=np.zeros((m,10)) #Might need to remove rows with NaN
    from numpy import nan
    for i in range(1,m):
        for j in range (3,n,4):
            p=int((j+1)/4-1)
            Qmat[i-1,p]=dat1.iat[i,j]+dat1.iat[i,j+1]
    Q=np.sum(Qmat)/(2*m*10)
    
    # Calculation of scaled OFI
    ofi=OFI/Q
    ofi_mat[:,jjk]=ofi  

# Removing entries that only have one element in the one-minute intervals
temp=pd.DataFrame(ofi_mat)
cols=temp.columns[temp.isnull().any()].tolist()
temp.drop(temp.columns[cols],axis=1,inplace=True)
ofi_mat=temp.to_numpy()

# Calculation of pricipal components
# First we normalize the columns
# Then we find the correlation matrix of the design matrix
# Then we find the eigenvectors and eigenvalues
ofi_mat=np.transpose(ofi_mat)
ofi_mean=np.mean(ofi_mat,axis=0)
ofi_std=np.std(ofi_mat,axis=0)
Z=ofi_mat-ofi_mean
Z=Z/ofi_std
n=temp.shape[1]
p=10
C=(1/(n-1))*np.matmul(np.transpose(Z),Z)
Eval,Evec=np.linalg.eig(C)
W=Evec[Eval==max(Eval)]
print("Here are the eigenvalues:")
print(Eval)
print("Here are the eigenvectors:")
print(Evec)
print("Here is the first principal vector:")
print(W)
print("Here is the percentage contribution of each of the principal component")
print((Eval/np.sum(Eval))*100)

# Calculation of integrated OFI
int_ofi=np.matmul(np.abs(W),np.transpose(ofi_mat))/np.sum(np.abs(W))
print(int_ofi)

#Storing the data in CSV files
temp.to_csv("AAPL_OFI.csv")
int_ofi_dat=pd.DataFrame(int_ofi)
int_ofi_dat.to_csv("AAPL_INT_OFI2.csv")

# Calculation of log returns
jt=len(price)
pt1=np.array(price[1:jt])
pt2=np.array(price[0:jt-1])
q=np.divide(pt1.astype('float'),pt2.astype('float'))
ret=np.log(q)
rt=pd.DataFrame(ret)
rt.drop(cols,axis=0,inplace=True)
rt.to_csv("AAPL_returns.csv")

#Storing the date and times 
uniq2=np.setdiff1d(uniq,uniq[cols])
uniq_dat=pd.DataFrame(uniq2)
uniq_dat.to_csv("AAPL_times.csv")
