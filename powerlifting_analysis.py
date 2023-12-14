import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import networkx as nx

# read
file_path = 'processed_powerlifting_data.csv'
data = pd.read_csv(file_path)

# normalize
scaler = StandardScaler()
features = ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'BodyweightKg']
data_scaled = scaler.fit_transform(data[features])

# k-means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# graph
G = nx.Graph()
for index, row in data.iterrows():
    node_attributes = {
        'Name': row['Name'],
        'BodyweightKg': row['BodyweightKg'],
        'Best3SquatKg': row['Best3SquatKg'],
        'Best3BenchKg': row['Best3BenchKg'],
        'Best3DeadliftKg': row['Best3DeadliftKg'],
        'Cluster': row['Cluster']
    }
    G.add_node(index, **node_attributes)

# is there meaningful way to connect nodes with edges?

# export
graphml_path = 'powerlifting_clusters.graphml'
nx.write_graphml(G, graphml_path)
