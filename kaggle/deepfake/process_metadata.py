import librosa
import warnings
warnings.filterwarnings('ignore')
import json
from tqdm import tqdm

#my_json = json.load('metadata.json')
with open('metadata.json','r') as f:
    data = json.loads(f.read())

for f in tqdm(data.keys()):
    f_now = f
    #print('running: ', f)
    info = data.get(f)
    f_ori = info.get('original')
    if f_ori != None:
        y_now,_ = librosa.load(f_now)
        y_ori,_ = librosa.load(f_ori)
        y_diff = y_now - y_ori
        sumup = y_diff.sum()
        info['diff'] = sumup
    #data[f] = info

# Serialize data into file:
json.dump( str(data), open( "file_name.json", 'w' ) )
# load
my_dict = eval(json.load( open( "file_name.json" ) ))

import json
data = json.load( open( "file_name.json" ) )

data = eval(data)
diff_data = dict()
for k in data.keys():
    info = data.get(k)
    diff = info.get('diff')
    if diff != 0.0:
        info['res'] = 'FAKE_DIFF'
        diff_data[k] = info
    else:
        info['res'] = 'FAKE_SAME'
