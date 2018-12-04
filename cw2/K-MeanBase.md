### K mean method for baseline learning

1. From the query, calculate how many cluster there is
2. Perform K-mean clustering on the gallery data
3. Project each query to check which cluster it belongs to 
4. Delete the same person from the same ID and Cam
5. Calculate the RANK of each query from the mean of the cluster
