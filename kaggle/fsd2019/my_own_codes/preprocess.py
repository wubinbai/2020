import pandas as pd
#TOT = 'input'
TOT = 'data'


#CURATED = '../' + TOT + '/audio_train/'
#NOISY = CURATED
#TEST = '../' + TOT + '/test/'

CURATED = '../' + TOT + '/train_curated/'
NOISY = '../' + TOT + '/train_noisy/'
TEST = '../' + TOT + '/test/'



curated = pd.read_csv('../' + TOT + '/train_curated.csv')
noisy = pd.read_csv('../' + TOT + '/train_noisy.csv')
test = pd.read_csv('../' + TOT + '/sample_submission.csv')

curated_paths = CURATED + curated['fname']
noisy_paths = NOISY + noisy['fname']
test_paths = TEST + test['fname']


def get_arr_classes(df):
    res = []
    count = 0
    for row in df.labels.values:
        split = row.split(',')
        num_labels = len(split)
        res.append(row.split(',')[0])
        count+=1
    res_df = pd.DataFrame(res)
    y0 = res_df = res_df[0] # Series.unique()
    res_df = res_df.unique()
    return res_df, y0


arr_classes_curated, y0_curated = get_arr_classes(curated)
arr_classes_noisy, y0_noisy = get_arr_classes(noisy)
#arr_classes_test = get_arr_classes(test)

# modified
curated_m = curated.copy()
noisy_m = noisy.copy()

curated_m['y0'] = y0_curated
noisy_m['y0'] = y0_noisy

