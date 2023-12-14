import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx

# read
file_path = 'processed_powerlifting_data.csv'
data = pd.read_csv(file_path)

# normalize
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'BodyweightKg']])

# plot graph using elbow method (wcss = within cluster sum of squares)
wcss = []
k_values = range(2, 6) 

for i in k_values:
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(k_values, wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
