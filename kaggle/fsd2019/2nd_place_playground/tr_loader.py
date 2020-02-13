from tqdm import tqdm
from sklearn.metrics import label_ranking_average_precision_score
from utils import *
from config import *

if True:
        df = pd.read_csv(f'../input/train_noisy.csv')
        y = split_and_label(df['labels'].values)
        x = train_dir + df['fname'].values
        x = [librosa.load(path, 44100)[0] for path in tqdm(x[:100])]
        x = [librosa.effects.trim(data)[0] for data in tqdm(x[:100])]

        gfeat = np.load('../input/gfeat.npy')[-len(x[:100]):]
       
                
        from models import *
        cfg = Config(
        duration=5,
        name='v1mix',
        lr=0.0005,
        batch_size=32,
        rnn_unit=128,
        momentum=0.85,
        mixup_prob=0.7,
        lm=0.01,
        pool_mode=('max', 'avemax1'),
        x1_rate=0.7,
        milestones=(8,12,16),
        get_backbone=get_conv_backbone)
        get_model = cnn_model

        K.clear_session()
        model = get_model(cfg)
        best_score = -np.inf
        epoch = 0
        

        tr_loader = FreeSound(x, gfeat, y, cfg, 'train', epoch)

