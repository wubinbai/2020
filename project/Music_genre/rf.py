from sklearn.ensemble import RandomForestClassifier


from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras import models
from keras import layers


df = pd.read_csv('data.csv')
data = df.copy()
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, 1:-1], dtype = float))





index = []
print('use 990+100 to apply last class')
for i in range(90,990,100):
    j = i + 10
    temp = list(range(i,j))
    index.extend(temp)

X_test, y_test = X[index], y[index]

t_i = train_index = []
print('use 900+100 to apply to last class')
for i in range(0,900,100):
    j = i + 90
    temp = list(range(i,j))
    train_index.extend(temp)
X_train, y_train = X[t_i], y[t_i]



### build a classifier
rf = RandomForestClassifier(n_estimators=600)
rf.fit(X_train,y_train)
preds = rf.predict(X_test)
for i in range(0,90,10):
    j = i+10
    print(preds[i:j])

