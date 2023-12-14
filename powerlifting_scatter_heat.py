import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv('processed_powerlifting_data.csv')

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
df['cluster'] = kmeans.fit_predict(df[['BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']])

# Scatter plot for Body Weight vs. Best Bench (Clustered)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='BodyweightKg', y='Best3BenchKg', hue='cluster', data=df, palette='Set1')
plt.title('Body Weight vs. Best Bench Press (Clustered)')
plt.xlabel('Body Weight (Kg)')
plt.ylabel('Best Bench Press (Kg)')
plt.show()

# Scatter plot for Body Weight vs. Best Squat (Clustered)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='BodyweightKg', y='Best3SquatKg', hue='cluster', data=df, palette='Set1')
plt.title('Body Weight vs. Best Squat (Clustered)')
plt.xlabel('Body Weight (Kg)')
plt.ylabel('Best Squat (Kg)')
plt.show()

# Scatter plot for Body Weight vs. Best Deadlift (Clustered)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='BodyweightKg', y='Best3DeadliftKg', hue='cluster', data=df, palette='Set1')
plt.title('Body Weight vs. Best Deadlift (Clustered)')
plt.xlabel('Body Weight (Kg)')
plt.ylabel('Best Deadlift (Kg)')
plt.show()

# Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df[['BodyweightKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlations')
plt.show()