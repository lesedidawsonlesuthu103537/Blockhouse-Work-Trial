import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb
import numpy as np

# Import file with data
data = pd.read_csv("XOM_OFI.csv") # To extract data for company ABC, enter "ABC_OFI.csv", whre ABC is one of TSLA, JPM, XOM, PFE and AAPL
dat=np.transpose(data)
print(dat.corr(numeric_only=True))
# Plotting correlation heatmap
dataplot = sb.heatmap(dat.corr(numeric_only=True), cmap="YlGnBu", annot=False)
# Displaying heatmap
mp.show()