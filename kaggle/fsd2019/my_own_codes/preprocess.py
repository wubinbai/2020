import pandas as pd
CURATED = '../input/audio_train/'
NOISY = CURATED
TEST = '../input/test/'

curated = pd.read_csv(f'../input/train_curated.csv')
noisy = pd.read_csv('../input/train_noisy.csv')
test = pd.read_csv('../input/sample_submission.csv')

curated_paths = CURATED + curated['fname']
noisy_paths = NOISY + noisy['fname']
test_paths = TEST + test['fname']


