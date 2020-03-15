df = pd.read_csv('submission.csv')
cols = df.columns[1:]
print(cols.shape)

m = 60
e = 120
for i in range(m,e):
    fname = df.loc[i].fname
    argmax_val = np.argmax(df.loc[i].values[1:])
    pred = df.loc[i].values[1:]
    sortted = np.argsort(pred)
    top1 = sortted[-1]
    assert top1 == argmax_val
    top2 = sortted[-2]
    top3 = sortted[-3]
    print(fname)
    print('top1:  ',cols[top1],'prob: ', pred[top1])
    print('top2:  ',cols[top2],'prob: ', pred[top2])
    print('top3:  ',cols[top3],'prob: ', pred[top3])

    #print(cols[ans])
    print('---')

