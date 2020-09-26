import shutil
from tqdm import tqdm
import os
import json
with open('../txt/soxi_test.txt') as f:
    data = f.readlines()
names = []
bitrates = []
for i in range(0,len(data)-1,10):
    name = data[1+i][:][-14:][:-2]
    names.append(name)
    bitrate = data[7+i][-6:][:-1]
    bitrates.append(bitrate)
arr_bitrates = np.array(bitrates)

dict_bitrates = dict()
for i in range(len(names)):
    dict_bitrates[names[i]] = bitrates[i]
with open('../test_dict_bitrates.json','w') as f:
    json.dump(dict_bitrates,f)


##### create folders and move wav into corresponding bitrate folder

os.makedirs('../eda/',exist_ok=True)
os.makedirs('../eda/test',exist_ok=True)
os.makedirs('../eda/test/256k',exist_ok=True)
os.makedirs('../eda/test/258k',exist_ok=True)
os.makedirs('../eda/test/259k',exist_ok=True)
os.makedirs('../eda/test/384k',exist_ok=True)
os.makedirs('../eda/test/512k',exist_ok=True)
os.makedirs('../eda/test/768k',exist_ok=True)
os.makedirs('../eda/test/1_06M',exist_ok=True)


for a,b in tqdm(dict_bitrates.items()):
    #print(a,b)
    src = '../test/' + a
    if not b.endswith('M'):
        dst = '../eda/test/' + b[1:]
    else:
        dst = '../eda/test/' + '1_06M'
    #print(src,dst)
    shutil.copy(src,dst)
