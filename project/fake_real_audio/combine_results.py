import json
import pandas as pd
blank_sub = pd.read_csv('../txt/blank_submission.txt',sep='\s+',names = 'Audio_Name Is_Faked Speaker_ID'.split(' '))
with open('../test_dict_bitrates.json','r') as f:
    dict_bitrates = json.load(f)
with open('../jsons/pred_256k_dict.json', 'r') as f:
    pred_256k_dict = json.load(f)
with open('../jsons/pred_768k_dict.json', 'r') as f:
    pred_768k_dict = json.load(f)




for x, y in dict_bitrates.items():
    if y == ' 512k' or y == ' 258k' or y == ' 259k':
        blank_sub.loc[blank_sub.loc[blank_sub['Audio_Name']==x].index[0],'Is_Faked'] = 1
    else:
        if y == ' 384k':
            blank_sub.loc[blank_sub.loc[blank_sub['Audio_Name']==x].index[0],'Speaker_ID'] = 'lmx_9714'
        if y == '1.06M':
            blank_sub.loc[blank_sub.loc[blank_sub['Audio_Name']==x].index[0],'Speaker_ID'] = 'ht_0145'
        if y == ' 256k':
            blank_sub.loc[blank_sub.loc[blank_sub['Audio_Name']==x].index[0],'Speaker_ID'] = pred_256k_dict[x]
        if y == ' 768k':
            blank_sub.loc[blank_sub.loc[blank_sub['Audio_Name']==x].index[0],'Speaker_ID'] = pred_768k_dict[x]

blank_sub.to_csv('../subs/combine_results.txt',sep=' ',header=False,index=False)
