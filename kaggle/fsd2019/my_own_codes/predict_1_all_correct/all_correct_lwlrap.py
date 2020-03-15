# you need following .py files

from my_metric import *
# use get_lwlrap(a,b) to return score!

from preprocess import *
# need this file


submit = pd.read_csv('./sample_submission.csv')
i2label = label_columns = submit.columns[1:].tolist()
label2i = {label:i for i,label in enumerate(i2label)}

n_classes = 80


def split_and_label(rows_labels):
    row_labels_list = []
    for row in rows_labels:
        row_labels = row.split(',')
        labels_array = np.zeros((n_classes))
        for label in row_labels:
            index = label2i[label]
            labels_array[index] = 1
        row_labels_list.append(labels_array)
    return np.array(row_labels_list)




truth_df = pd.read_csv('train_curated.csv')
pred_df = curated_m

truth = split_and_label(truth_df.labels)
pred = split_and_label(pred_df.y0)
# what is only 100 rows
#truth = truth[:100,:]
#pred = pred[:100,:]
#x = list(range(0,80))
#xx = [h/80 for h in x]
#pred[:40,:] = 0

#print(pred.shape)
#ans = get_lwlrap(pred,truth)
#print(ans)



ans = calculate_overall_lwlrap_sklearn(truth,pred)
print('1.0 acc: ', ans)



# what is 0.9 only are correctly predicted
pred[0:int(4970*0.1),:] = 0
ans = calculate_overall_lwlrap_sklearn(truth,pred)
print('0.9 acc: ', ans)


# what is 0.8 only are correctly predicted
pred[0:int(4970*0.2),:] = 0
ans = calculate_overall_lwlrap_sklearn(truth,pred)
print('0.8 acc: ', ans)


# what is 0.7 only are correctly predicted
pred[0:int(4970*0.3),:] = 0
ans = calculate_overall_lwlrap_sklearn(truth,pred)
print('0.7 acc: ', ans)


# what is 0.6 only are correctly predicted
pred[0:1988,:] = 0
ans = calculate_overall_lwlrap_sklearn(truth,pred)
print('0.6 acc: ', ans)

