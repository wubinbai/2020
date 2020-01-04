
import pandas as pd
from rmse import rmse_cv
from sklearn.linear_model import Ridge, Lasso, LassoCV
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
corr = train.corr()
y_train = train.SalePrice
train.drop('SalePrice',axis=1,inplace=True)
y_train_transformed = np.log1p(y_train)
test_id = test.Id

useless = ['Utilities', 'BsmtFinType2','BsmtFinSF2','BsmtUnfSF','HalfBath','YrSold']
DROP_USELESS = True
def drop_useless(df):
    print('before drop_useless: df has shape: ', df.shape)
    for i in useless:
        df.drop(i,axis=1,inplace=True)
    print('after drop_useless: df has shape: ', df.shape)

if DROP_USELESS:
    drop_useless(train)
    drop_useless(test)
LEVEL = 0
if LEVEL == 0:
    train_test = pd.concat([train,test],axis=0)
    train_test_dummies = pd.get_dummies(train_test)
    train_d = train_dummies = train_test_dummies[:train.shape[0]]
    test_d = test_dummies = train_test_dummies[train.shape[0]:]
    train_fill = train_d.fillna(train_d.mean())
    test_fill = test_d.fillna(test_d.mean())

LEVEL2 = True
if LEVEL2:
    ss0 = StandardScaler()
    train_fill = pd.DataFrame(ss0.fit_transform(train_fill))
    ss1 = StandardScaler()
    test_fill = pd.DataFrame(ss1.fit_transform(test_fill))
#    tr0 = pd.get_dummies(train)
#    tr1 = tr0.fillna(tr0.mean())
#    te0 = pd.get_dummies(test)
#    te1 = te0.fillna(te0.mean())

alpha_ridge = [0.01, 0.03, 0.1, 0.3, 1, 3, 5, 7, 9, 10, 11,12, 13,15, 20, 30, 40, 50, 60, 70, 80]
model_ridge = [Ridge(alpha=alpha_i) for alpha_i in alpha_ridge]
cv_ridge = [rmse_cv(j, train_fill, y_train_transformed).mean() for j in model_ridge]
scores_ridge = cv_ridge
#scores_ridge = rmse_cv(model_ridge,train_fill,y_train_transformed)
print('scores_ridge: ', scores_ridge)
index_alpha_ridge = cv_ridge.index(min(cv_ridge))
best_alpha_ridge = alpha_ridge[index_alpha_ridge]
best_model_ridge = model_ridge[index_alpha_ridge]
best_model_ridge.fit(train_fill,y_train_transformed)
ridge_pred = best_model_ridge.predict(test_fill)

ridge_pred_transformed = np.expm1(ridge_pred)
ridge_dict = {'id':test_id, 'SalePrice':ridge_pred_transformed}
ridge_df = pd.DataFrame(ridge_dict)
ridge_df.to_csv('ridge_sub.csv',index=False)


# Same codes apply to Lasso below: but alphas should be different SMALLER!

alpha_lasso = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
# [0.01, 0.03, 0.1, 0.3, 1, 3, 5, 7, 9, 10, 11,12, 13,15, 20, 30, 40, 50, 60, 70, 80]
model_lasso = [Lasso(alpha=alpha_i) for alpha_i in alpha_lasso]
cv_lasso = [rmse_cv(j, train_fill, y_train_transformed).mean() for j in model_lasso]
scores_lasso = cv_lasso
print('scores_lasso: ', scores_lasso)
index_alpha_lasso = cv_lasso.index(min(cv_lasso))
best_alpha_lasso = alpha_lasso[index_alpha_lasso]
best_model_lasso = model_lasso[index_alpha_lasso]
best_model_lasso.fit(train_fill,y_train_transformed)
lasso_pred = best_model_lasso.predict(test_fill)


lasso_pred_transformed = np.expm1(lasso_pred)
lasso_dict = {'id':test_id, 'SalePrice':lasso_pred_transformed}
lasso_df = pd.DataFrame(lasso_dict)
lasso_df.to_csv('lasso_sub.csv',index=False)




dtrain = xgb.DMatrix(train_fill, label=y_train)
dtest = xgb.DMatrix(test_fill)
params = {"max_depth":2, "eta":0.1}
xgb_model_df = xgb.cv(params, dtrain, num_boost_round=500, early_stopping_rounds=100)
xgb_model_df.loc[30:,['test-rmse-mean','train-rmse-mean']].plot()

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)
model_xgb.fit(train_fill,y_train_transformed)
xgb_pred = model_xgb.predict(test_fill)
xgb_pred_transformed = np.expm1(xgb_pred)

xgb_dict = {'id':test_id, 'SalePrice':xgb_pred_transformed}
xgb_df = pd.DataFrame(xgb_dict)
xgb_df.to_csv('xgb_sub.csv',index=False)

