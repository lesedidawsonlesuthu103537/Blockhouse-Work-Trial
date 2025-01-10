# Scripts
This folder contains the scripts used to clean the data obtained from the MBP-10 data on the various companies analysed and store dataframes as csv files, containing data that will be used in either the regression models or the correlation heatmaps.

The inputs to these files are the "COM_raw_data.csv", which is a dataset that contains the raw limit order book data of company COM obtained from the Databento database. The code contains comments to help you navigate the code and understand what functions are being performed at various sections. The outputs of these scripts include:

* Integrated OFI metrics for all minute points considered on 19 August 2024
* OFI metrics for the top 10 levels
* Logarithmic returns
* The times at which the minute OFI, integrated OFI and logarithmic returns where calculated
