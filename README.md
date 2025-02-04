# Blockhouse Work Trial
The purpose of this project was to evaluate cross-asset impacts on short-term price changes. To do this, the contemporaneous returns of each stock under investigation were regressed against the best-level OFI and integrated OFI (Order Flow Imbalance) metrics of all companies under investigation. This was done to determine the effect of the self-impact and cross-impact terms on mid-price changes. Once this analysis was done, the OFI metrics were regressed against future returns to determine the ability of the OFI to forecast short-term price changes.

This repository contains the data and the python scripts to conduct this investigation. To be able to replicate the results in the report, you must: 

* First download the datasets in the link attached in the README in the data folder. In particular, you must download all the "COM_raw_data.csv" files for each company COM.
* Secondly, you must run the "COM.py" scripts from the "Data cleaning and OFI calculation" subfolder in the "scripts" folder. This will output the remaining files in the link mentioned in the first bullet point
* Thirdly, ensure that the output files from the second bullet point are all stored in the same working directory and save the "Corr.py" and "Reg.py" scripts from the "Correlation plots and regression model" subfolder in the "scripts" folder in the same directory.
* Fourthly, you can run the "Corr.py" and "Reg.py" to obtain the correlation heatmaps and regression results found in the "results" folder.

Here is a summary of the key findings of this work trial

* Both best-level and integrated OFIs, the cross impact has approximately one-tenth of the impact of the self/price impact in estimating contemporaneous returns.
* The integrated OFI outperforms the best-level OFI when it comes to explaining contemporaneous returns due to the impact of heavy inflows/outflows of orders on the buy and sell side on the mid-price, as it is a signal of heavy buying or selling pressure.
* When it comes to predicting future returns, both the self and the cross impact have a small and negative impact on future returns.
* This can possibly be attributed to the price pressure caused by the autocorrelation of imbalances diminishing very quickly in a high-frequency trading environment.
