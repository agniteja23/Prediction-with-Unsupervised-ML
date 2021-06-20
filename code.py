# Importing all libraries required in this notebook
import numpy as np #For numerical operations
import pandas as pd #For handling the dataset
import matplotlib.pyplot as plt #For visualization
import seaborn as sns
%matplotlib inline
df = pd.read_csv(r'Iris.csv')
df
df.drop('Id',axis=1,inplace=True)
df.head()
df.isnull().sum()
df.info()
df.describe()
#Finding the correlation between the features
df.corr()
# Checking outliers
sns.boxplot(x=df['SepalLengthCm'])
# Visualizing the correlation between the features
sns.heatmap(df.corr())
# Visualizing the correlation between the features
sns.set_style("darkgrid")
sns.pairplot(df, hue="Species", height=3);
plt.show()
from sklearn.cluster import KMeans
ks = range(1, 6)
inertias = []
x = df.iloc[:, [0, 1, 2, 3]].values
for k in ks:
    
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(x)
    
   
 # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o',linestyle='dashed',color='green')
plt.title('The elbow method')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
kmeans = KMeans(n_clusters=3)

kmeans.fit(x)
# Calculate the cluster labels: labels
labels = kmeans.predict(x)

# Visualising the clusters - On the first two columns
plt.scatter(x[labels == 0, 0], x[labels == 0, 1], 
            s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(x[labels == 1, 0], x[labels == 1, 1], 
            s = 100, c = 'violet', label = 'Iris-versicolour')
plt.scatter(x[labels == 2, 0], x[labels == 2, 1],
            s = 100, c = 'red', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'black', label = 'Centroids')

plt.legend()
# Create a DataFrame with labels and species as columns: df
df2 = pd.DataFrame({'labels':labels,'species':df['Species']})

# Create crosstab: ct
ct = pd.crosstab(df2['labels'],df2['species'])

print(ct)
# Perform the necessary imports
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = Normalizer()

x = scaler.fit_transform(x)

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=3)

# Create pipeline: pipeline
kmeans.fit(x)
# Calculate the cluster labels: labels
labels = kmeans.predict(x)

# Visualising the clusters - On the first two columns
plt.scatter(x[labels == 0, 0], x[labels == 0, 1], 
            s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(x[labels == 1, 0], x[labels == 1, 1], 
            s = 100, c = 'violet', label = 'Iris-versicolour')
plt.scatter(x[labels == 2, 0], x[labels == 2, 1],
            s = 100, c = 'red', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'black', label = 'Centroids')

plt.legend()
# Create a DataFrame with labels and species as columns: df
df2 = pd.DataFrame({'labels':labels,'species':df['Species']})

# Create crosstab: ct
ct = pd.crosstab(df2['labels'],df2['species'])

# Display ct
print(ct)
