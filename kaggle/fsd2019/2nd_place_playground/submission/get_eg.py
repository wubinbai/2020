sub = pdrc('submission.csv')


In [88]: def get_eg():  
    ...:     number = int(input('Enter number of instance: 
    ...: '))
    ...:     pred_number = np.argmax(sub.loc[number,:].valu
    ...: es[1:])
    ...:     name = sub.columns[1:][pred_number]
    ...:     filename = sub.fname[number]
    ...:     print('the file name is: ', filename)
    ...:     print('predicted as: ', name)
    ...:     

