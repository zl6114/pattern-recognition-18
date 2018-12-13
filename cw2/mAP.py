average_precision=[]
Q_ranklist = np.loadtxt(open("query_rank_list.csv", "rb"), delimiter=",")
for i in tqdm_notebook(range(len(Q_ranklist)),miniters=10):
    Precision=[]
    Recall=[]
    precisions=[]
    senses=sum(Q_ranklist[i,:])
    for j in range(1,Q_ranklist.shape[1]):
        Precision.append(sum(Q_ranklist[i,:j]/j))
        Recall.append(sum(Q_ranklist[i,:j]/senses))
        if Recall[j-1] == 1:
              break

    u=[]
    indices=[]
    Recall=np.asarray(Recall)
    Precision=np.asarray(Precision)
    u, indices=np.unique(Recall,return_index=True)

    precisions=Precision[indices]
    precisions=precisions[precisions!=0]
    average_precision.append(np.mean(precisions))
average_precision = np.nan_to_num(average_precision)
print(np.asarray(average_precision))    
print(np.mean(average_precision[1:]))
