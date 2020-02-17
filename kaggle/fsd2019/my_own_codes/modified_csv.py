import pandas as pd
TOT = 'data'
CURATED = '../' + TOT + '/train_curated/'
NOISY = '../' + TOT + '/train_noisy/'
TEST = '../' + TOT + '/test/'

curated = pd.read_csv('../' + TOT + '/train_curated.csv')
noisy = pd.read_csv('../' + TOT + '/train_noisy.csv')
test = pd.read_csv('../' + TOT + '/sample_submission.csv')

curated_paths = CURATED + curated['fname']
noisy_paths = NOISY + noisy['fname']
test_paths = TEST + test['fname']


def get_max_label(df):
    max_label = 1
    for label in df.labels.values:
        splitted = label.split(',')
        if len(splitted) >= max_label:
            max_label = len(splitted)
    return max_label
max_label_curated = get_max_label(curated)
max_label_noisy = get_max_label(noisy)

def modify_csv(df):

    #### initialize modified
    modified = df.copy()
    max_label = get_max_label(df)
    for i in list(range(max_label)):
        name = 'y' + str(i)
        modified[name] = 'Nothing'
    #### finish initialize modified
    count = 0
    for label in df.labels.values:
        splitted = label.split(',')
        length = len(splitted)
        for i in range(length):
            temp = 'y' + str(i)
            modified.loc[count][temp] = splitted[i]
        count += 1

    return modified

curated_m = modify_csv(curated)
noisy_m = modify_csv(noisy)

def collect_multi_labels(df):
    res = []
    curated_m = modify_csv(df)
    for i in range(1,get_max_label(df)):
        temp = 'y' + str(i)
        want = curated_m[curated_m.loc[:,temp] != 'Nothing']
        # MODIFY WANT for difference of set
        res.append(want)
    return res

multi_labels_curated = collect_multi_labels(curated)
multi_labels_noisy = collect_multi_labels(noisy)
