import json
orig_sub = pdrc('../subs/sub_yong.txt',sep='\s+',names='Audio_Name IsFaked ID'.split(' '))
with open('../test_dict_bitrates.json','r') as f:
    dict_bitrates = json.load(f)
count = 0
modified = 0
for x,y in dict_bitrates.items():
    #print(x,y)
    
    if y == ' 512k' or y == ' 258k' or y == ' 259k':
        before = orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name'] == x].index[0],'IsFaked']
        orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name']==x].index[0],'IsFaked'] = 1
        after = orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name'] == x].index[0],'IsFaked']
        count +=1
        if before != after:
            modified += 1
            print(before, '-----> ', after)
    if y == ' 768k':
        before = orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name'] == x].index[0],'IsFaked']
        orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name']==x].index[0],'IsFaked'] = 0
        after = orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name'] == x].index[0],'IsFaked']
        count +=1
        if before != after:
            modified += 1
            print(before, '-----> ', after)

    if y == ' 384k':
        before = orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name']==x].index[0],'IsFaked'] 
        orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name']==x].index[0],'IsFaked'] = 0
        after = orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name']==x].index[0],'IsFaked'] 
        if not before == after:
            print(before,'--->',after)

        before = orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name']==x].index[0],'ID']
        orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name']==x].index[0],'ID'] = 'lmx_9714'
        after = orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name']==x].index[0],'ID']
        if not before == after:
            print(before, '--->',after)

    if y == '1.06M':
        before = orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name']==x].index[0],'IsFaked'] 
        orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name']==x].index[0],'IsFaked'] = 0
        after = orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name']==x].index[0],'IsFaked'] 
        if not before == after:
            print(before,'--->',after)


        before = orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name']==x].index[0],'ID']
        orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name']==x].index[0],'ID'] = 'ht_0145'
        after = orig_sub.loc[orig_sub.loc[orig_sub['Audio_Name']==x].index[0],'ID']
        if not before == after:
            print(before, '--->',after)




print('modified ', modified, 'out of ', count)
orig_sub.to_csv('../subs/modified_sub_yong.txt',sep=' ',header=False,index=False)
