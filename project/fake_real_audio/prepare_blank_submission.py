import os
import pandas as pd

test_fs = os.listdir('../test')
fake_real = [0] * len(test_fs)
ids = [0] * len(test_fs)
d = {'Audio_Name':test_fs,'Is_Faked':fake_real,'Speaker_ID':ids}
df = pd.DataFrame(d)
df.to_csv('../txt/blank_submission.txt',sep=' ',index=False,header=False)
