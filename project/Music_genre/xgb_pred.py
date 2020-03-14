import librosa
import os
from tqdm import tqdm
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler



def train_and_eval(n_feats_now, to_save='n'):
    # save all feature data
    if to_save == 'y':
        save_data(n_feats_now)
    else:
        pass
    data = get_data()
    # load only data of feature chosen
    Xy = data.iloc[:,1:] # drop name column
    # get X( only :n_feats_now) and y
    X = Xy.iloc[:,:n_feats_now]
    y = Xy.iloc[:,-1]
    
    # for non-tree, like knn, scale it!
    #sc = StandardScaler()
    #X = sc.fit_transform(X)
    #X = pd.DataFrame(X)

    # label encode y
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    y = pd.DataFrame(y)
    # fit data
    # stratified kfold
    sfolder = StratifiedKFold(n_splits=10,random_state=0,shuffle=False)
    ### build a classifier
    #knn = KNeighborsClassifier()
    model = XGBClassifier()
    #rf = RandomForestClassifier(n_estimators=800)
    print('For debug: initiating model')
    # using stratifiedkfold 
    scores = []
    for train_index, test_index in sfolder.split(y,y):
        X_train = X.loc[train_index]
        X_test = X.loc[test_index]
        y_train = y.loc[train_index]
        y_test = y.loc[test_index]
        #print('shape of X_train: ',X_train.shape)
        #print('shape of y_train: ',y_train.shape)
        #print(y_train.head())
        model.fit(X_train,y_train)
        preds = model.predict(X_test)
        for i in range(0,100,10):
            j = i+10
            print(preds[i:j])

        all_correct = 0
        for i in range(int(preds.shape[0]/10)):
            ans = i
            temp = preds[i*10:i*10+10]
            correct = (temp == ans)
            correct = correct.sum()
            all_correct += correct

        score = all_correct / preds.shape[0]
        scores.append(score)

    #plt.figure()
    #plt.plot(knn.feature_importances_)
    #plt.plot(knn.feature_importances_,'r*')
    #plt.savefig('feat_imp.png')
    # get score
    score = round(np.mean(scores),4)
    print('cross_val score: ', score)
    #### finish stratified kfold
    return score

def get_data():
    df = pd.read_csv('saved_data.csv')
    print(df.shape)
    return df
 

def run_num_feats(n_feats,tosave):
    #for i in range(1,n_feats+1):
        n_feats_now = n_feats
        print('running model with ', n_feats_now, 'feats')
        score = train_and_eval(n_feats_now,to_save=tosave)
        print('the score for ', n_feats_now, 'is', score)
        return score
if __name__ == '__main__':
    #n_feats = 2
    #run_num_feats(n_feats)
    #save_data(2)
    #get_data()
    all_scores = []
    start_feat = int(input('Enter starting num feats to run: '))
    num = int(input('Enter num feats to run: '))
    first_time = 0
    for i in range(start_feat,num+1):
        if first_time !=0:
            my_save = 'n'
        if first_time ==0:
            my_save = input('Enter y or n for save: ')
        first_time+=1
        score = run_num_feats(i,tosave=my_save)
        all_scores.append(score)
    #pass
