# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries, load the mall customer dataset, and check for missing values and dataset information.
2. Apply the K-Means clustering algorithm with different cluster values to calculate WCSS and use the Elbow Method to find the optimal number of clusters.
3. Train the K-Means model with 5 clusters, predict customer groups, and store the cluster labels in the dataset.
4. Separate the customers based on clusters and visualize the customer segments using a scatter plot of Annual Income vs Spending Score.


## Program:
```
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: ROUNAK SINGH
RegisterNumber:  212225240125
```
```PY
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")
data.head()
data.info()
data.isnull()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss= [] 
for i in range(1,11):
kmeans=KMeans(n_clusters = i,init = "k-means++")
kmeans.fit(data.iloc[:,3:])
wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
km=KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])
y_pred=km.predict(data.iloc[:,3:])
y_pred
data["cluster"]=y_pred
1/4
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="black",label="clu
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="cyan",label="clust
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="yellow",label="cl
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="blue",label="clust
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="green",label="clu
plt.legend()
plt.title("Customer Segments")

```
## Output:
<img width="857" height="296" alt="image" src="https://github.com/user-attachments/assets/f16e5c4f-5e00-4a7b-a6e1-0b9699d119cd" />

<img width="948" height="777" alt="image" src="https://github.com/user-attachments/assets/e4dc5563-d140-4a7c-ad7d-9274bcc1e940" />

<img width="647" height="527" alt="image" src="https://github.com/user-attachments/assets/5f2e6f7e-1d25-4ce7-b20f-9bec0c877cca" />

<img width="855" height="272" alt="image" src="https://github.com/user-attachments/assets/165faf7b-667e-4309-a1a8-55d0fc74912b" />

<img width="682" height="395" alt="image" src="https://github.com/user-attachments/assets/c91c2be8-2e3e-4ff0-bac9-b2a46b1e5191" />

## Result:

The program was successfully excecuted and run on system
