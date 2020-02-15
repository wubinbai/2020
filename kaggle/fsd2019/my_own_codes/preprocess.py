import pandas as pd
CURATED = '../input/audio_train/'
NOISY = CURATED
TEST = '../input/test/'

#TOT = 'input'
TOT = 'data'
curated = pd.read_csv('../' + TOT +'/train_curated.csv')
noisy = pd.read_csv('../' + TOT +'/train_noisy.csv')
test = pd.read_csv('../' + TOT +'/sample_submission.csv')

curated_paths = CURATED + curated['fname']
noisy_paths = NOISY + noisy['fname']
test_paths = TEST + test['fname']


def get_arr_classes(df):
    res = []
    for row in df.labels.values:
        res.append(row.split(',')[0])
    res_df = pd.DataFrame(res)
    s = res_df = res_df[0] # Series.unique()
    res_df = res_df.unique()
    return res_df, s


arr_classes_curated, s_curated = get_arr_classes(curated)
arr_classes_noisy, s_noisy = get_arr_classes(noisy)
#arr_classes_test = get_arr_classes(test)

# modified
curated_m = curated.copy()
noisy_m = noisy.copy()

curated_m['y0'] = s_curated
noisy_m['y0'] = s_noisy

