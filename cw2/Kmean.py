from sklearn.cluster import KMeans
#size(n_samples, n_features)
kmeans = KMeans(n_clusters=np.size(np.unique(query_label)), random_state=0).fit(gallery_set)
kmeans.cluster_centers_
kmeans.predict
print(np.size(kmeans.labels_))
print(np.size(kmeans.cluster_centers_)/2048)
print(np.size(kmeans.cluster_centers_[110]))
predicted_class = kmeans.predict(query_set)
print(predicted_class.size)
kmeans.labels_.tolist().index(101)
searchval = 0
ii = np.where(kmeans.labels_ == searchval)[0]
print(ii)
from joblib import dump, load
dump(kmeans, 'kmeans.joblib') 
# Load from file
kmeans = load('kmeans.joblib')

predicted_class = kmeans.predict(query_set)
# Query Augmented
qs = predicted_class.T
query_augmented = np.vstack( ( qs, query_camId, query_label ) )
query_augmented = query_augmented.T

qs = kmeans.labels_.T
gallery_augmented = np.vstack( ( qs, gallery_camId, gallery_label ) )
gallery_augmented = gallery_augmented.T

query_rank_list_kmean = []

# for i in range( 2,4 ):
for i in tqdm_notebook(range(predicted_class.shape[0])): 
    
    ii = np.where(kmeans.labels_ == predicted_class[i])[0]
    for j in ii:
        if (gallery_augmented[j,1] != query_augmented[i,1]):
            query_rank_list_kmean.append(gallery_augmented[j,2] == query_augmented[i,2])  
    
query_rank_list_kmean = np.asarray( query_rank_list_kmean )

print(sum(query_rank_list_kmean)/query_rank_list_kmean.shape[0])
