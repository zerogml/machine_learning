import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
color = sns.color_palette()
import xgboost as xgb
import jieba.posseg as pseg
from sklearn import preprocessing
import jieba
import re
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
import random
from sklearn.metrics import fbeta_score, make_scorer
import fasttext
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn import linear_model
import joblib


# 字符串清洗，去除停用词
def clean_str(stri):
    stri = re.sub(u'[\s]+|[^\u4e00-\u9fa5A-Za-z]+|', '', stri)
    # stri = re.sub(r'|[\s+\.\!\/_\-,$%^*(+\"\']+|[+—【】！，。？、～~@#￥%……&*（）]|[0-9]+', ' ', stri)
    cut_str = jieba.cut(stri.strip())
    list_str = [word for word in cut_str if word not in stop_word]
    stri = ' '.join(list_str)
    return stri

# 空白的处理方式
def fillnull(x):
    if x == '':
        return '_na_'
    else:
        return x


# 构造fasttext使用的文本
def fasttext_data(data, label):
    fasttext_data = []
    for i in range(len(label)):
        sent = data[i] + "\t__label__" + str(int(label[i]))
        fasttext_data.append(sent)
    with open('train.txt', 'w') as f:
        for data in fasttext_data:
            f.write(data)
            f.write('\n')
    return 'train.txt'


# 得到预测值
def get_predict(pred):
    if pred.shape[1] == 5:
        score = np.array([1, 2, 3, 4, 5])
    if pred.shape[1] == 4:
        score = np.array([1, 2, 3, 4])
    if pred.shape[1] == 3:
        score = np.array([1, 2, 3])
    if pred.shape[1] == 2:
        score = np.array([1, 2])
    if pred.shape[1] == 1:
        score = np.array([1])
    pred2 = []
    for p in pred:
        pr = np.sum(p*score)
        pred2.append(pr)
    return np.array(pred2)


# 评测函数
def rmsel(true_label, pred):
    true_label = np.array(true_label)
    pred = np.array(pred)
    n = len(true_label)
    a = true_label - pred
    rmse = np.sqrt(np.sum(a * a) / n)
    b = 1 / (1 + rmse)
    return b


# 交叉检验
def lrnb_cv(model1, model2, model3, df, test_df, train_merge):
    df = df.sample(frac=1)  # 对行做shuffle
    df = df.reset_index(drop=True)

    # 取出模型，lr_model和nb_model
    nb_model = model1
    lr_model = model2
    ri_model = model3
    X = trn_term_doc_scale
    y = df['Score'].values
    lr_pred, nb_pred, ri_pred = [], [], []
    folds = list(KFold(n_splits=5, shuffle=True, random_state=2018).split(X, y))

    for train_index, test_index in folds:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 朴素贝叶斯训练
        nb_model.fit(X_train, y_train)
        pred_i = nb_model.predict_proba(X_test)
        pred_i = get_predict(pred_i)
        print('nb cv:', rmsel(y_test, pred_i))
        train_merge.loc[test_index, 'nb'] = pred_i  # 将验证集nb预测值进行存储
        train_merge.loc[test_index, 'score1'] = y_test  # 将验证集实际结果进行存储


        # 逻辑回归训练
        lr_model.fit(X_train, y_train)
        pred_i = lr_model.predict_proba(X_test)
        pred_i = get_predict(pred_i)
        print('lr cv:', rmsel(y_test, pred_i))
        train_merge.loc[test_index, 'lr'] = pred_i  # 将验证集lr预测值进行存储
        train_merge.loc[test_index, 'score2'] = y_test  # 将验证集实际结果进行存储

        # 岭回归训练
        ri_model.fit(X_train, y_train)
        pred_i = ri_model.predict(X_test)
        print('ri cv:', rmsel(y_test, pred_i))
        train_merge.loc[test_index, 'ri'] = pred_i  # 将验证集ridge预测值进行存储
        train_merge.loc[test_index, 'score4'] = y_test  # 将验证集实际结果进行存储

        # 朴素贝叶斯预测
        nb_predi = nb_model.predict_proba(test_term_doc_scale)
        nb_predi = get_predict(nb_predi)
        nb_pred.append(nb_predi)

        # 逻辑回归预测
        lr_predi = lr_model.predict_proba(test_term_doc_scale)
        lr_predi = get_predict(lr_predi)
        lr_pred.append(lr_predi)

        # 岭回归预测
        ri_predi = ri_model.predict(test_term_doc_scale)
        ri_pred.append(ri_predi)

    nb_pred = np.array(nb_pred)
    nb_pred = np.mean(nb_pred, axis=0)

    lr_pred = np.array(lr_pred)
    lr_pred = np.mean(lr_pred, axis=0)

    ri_pred = np.array(ri_pred)
    ri_pred = np.mean(ri_pred, axis=0)
    return nb_pred, lr_pred, ri_pred  # 返回三个模型预测结果


# fasttext模型
def fast_cv(df, test_df, train_merge):
    df = df.sample(frac=1,random_state=2018)  # 对行做shuffle
    df = df.reset_index(drop=True)

    fast_pred = []
    folds = list(KFold(n_splits=5, shuffle=True, random_state=2018).split(X, y))
    rmsels = []
    for train_index, test_index in folds:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_file = fasttext_data(X_train, y_train)
        # fasttext训练
        classifier = fasttext.train_supervised(train_file,lr=0.08, dim=256, word_ngrams=3, bucket=200000,
                                         loss='hs', label_prefix="__label__")
        result = classifier.predict(df.loc[test_index, 'Discuss'].tolist(), k=5)
        i = len(np.array(result[1]))  # 行数
        j = len(np.array(result[1])[0])  # 列数
        for l in range(i):
            for m in range(len(np.array(result[1])[l])):
                np.array(result[1])[l][m] = (5 - m) * np.array(result[1])[l][m]
        pred = np.array(result[1])
        pred = [sum(pred_i) for pred_i in pred]

        print('fast cv:', rmsel(y_test, pred))
        train_merge.loc[test_index, 'fast'] = pred  # 将验证集fasttext预测值进行存储
        train_merge.loc[test_index, 'score3'] = y_test  # 将验证集实际结果进行存储

        # fasttext预测
        test_result = classifier.predict(test_df['Discuss'].tolist(), k=5)

        i = len(np.array(test_result[1]))  # 行数
        j = len(np.array(test_result[1])[0])  # 列数
        for l in range(i):
            for m in range(len(np.array(test_result[1])[l])):
                np.array(test_result[1])[l][m] = (5 - m) * np.array(test_result[1])[l][m]
        fast_predi = np.array(test_result[1])
        fast_predi = [sum(pred_i) for pred_i in fast_predi]
        fast_pred.append(fast_predi)
    fast_pred = np.array(fast_pred)
    fast_pred = np.mean(fast_pred, axis=0)
    return fast_pred  # 返回fasttext模型预测结果


def modelfit(alg, X, y, useTrainCV=True, early_stopping_rounds=50, cv_folds=5, printFeatureImportance=True):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X, y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='rmse', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print(cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(X, y, eval_metric='rmse')

    # Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()


# 读取数据集
data_path = 'data2/train_second.csv'
df = pd.read_csv(data_path, header=0)

df2 = pd.read_csv('data2/train_first.csv', header=0)
df = pd.concat([df, df2], ignore_index=True)

test_data_path = 'data2/predict_first.csv'
test_df = pd.read_csv(test_data_path, header=0)

# 训练集去重
df.drop_duplicates(subset='Discuss', keep='last', inplace=True)

# 加载停用词
stop_word = []
stop_words_path = 'dict/stopWordList.txt'

with open(stop_words_path, encoding='utf8') as f:
    for line in f.readlines():
        stop_word.append(line.strip())
stop_word.append(' ')


# 加载情感词
dict_path = 'dict/dict.txt'
jieba.load_userdict(dict_path)

#  去除停用词
df['Discuss'] = df['Discuss'].map(lambda x: clean_str(x))
test_df['Discuss'] = test_df['Discuss'].map(lambda x: clean_str(x))

# 处理去除停用词后的空白部分 填充na
df['Discuss'] = df['Discuss'].map(lambda x: fillnull(x))
test_df['Discuss'] = test_df['Discuss'].map(lambda x: fillnull(x))


# tf-idf
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.8, use_idf=1, smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(df['Discuss'])
test_term_doc = vec.transform(test_df['Discuss'])

# print(type(test_term_doc))
# tf-idf降维
tsvd = TruncatedSVD(n_components=180)
tsvd.fit(trn_term_doc)
trn_term_doc = tsvd.transform(trn_term_doc)
test_term_doc = tsvd.transform(test_term_doc)

min_max = preprocessing.MinMaxScaler(feature_range=(0, 1))
trn_term_doc_scale = min_max.fit_transform(trn_term_doc)
test_term_doc_scale = min_max.transform(test_term_doc)


# 融合模型
nb_model = MultinomialNB()  # 朴素贝叶斯回归
lr_model = LogisticRegression(C=10, class_weight='balanced')  # 逻辑回归模型
ri_model = linear_model.Ridge()  # 岭回归模型

# 加载三个模型结果
data = np.zeros((len(df), 8))
train_merge = pd.DataFrame(data)
train_merge.columns = ['nb', 'lr', 'fast', 'ri', 'score1', 'score2', 'score3', 'score4']
nb_pred, lr_pred, ri_pred = lrnb_cv(nb_model, lr_model, ri_model, df, test_df, train_merge)

# fasttext模型结果
X = df['Discuss'].values
y = df['Score'].values
fast_pred = fast_cv(df, test_df, train_merge)


# 创建测试集
data = np.zeros((len(test_df), 4))
test = pd.DataFrame(data)
feature_columns=['nb','lr', 'fast','ri']
test.columns = ['nb','lr', 'fast','ri']
test['nb'], test['lr'], test['fast'], test['ri'] = nb_pred, lr_pred, fast_pred, ri_pred
test.describe()

# xgb调参
# 数据准备
X = train_merge[feature_columns].values
y = train_merge['score1'].values

# 得到学习速率 为0.1时的理想决策树目
xgb1 = xgb.XGBRegressor(learning_rate=0.1,n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,
                      objective='reg:linear', eval_metric='rmse', scale_pos_weight=1, seed=2018)  # 回归

modelfit(xgb1, X, y)   # 分类  训练 预测


score = make_scorer(rmsel)
params_test1 = {'max_depth': list(range(3,8,2)), 'min_child_weight': list(range(1,6,2))}
xgb2 = xgb.XGBRegressor(learning_rate=0.1,n_estimators=110, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,
                      objective='reg:linear', eval_metric='rmse', scale_pos_weight=1, seed=2018)
#  网格搜索调优
gsearch1 = GridSearchCV(estimator=xgb2, param_grid=params_test1, scoring=score, cv=5)
gsearch1.fit(X, y)

gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_

#  最终的最优模型
bst = xgb.XGBRegressor(learning_rate=0.01, n_estimators=1100, max_depth=5, min_child_weight=5, gamma=0, subsample=0.8,
                       eval_metric='rmse', scale_pos_weight=1, seed=2018)

xgb_pred = []
folds = list(KFold(n_splits=5, shuffle=True, random_state=2018).split(X, y))
es = []
for tr_index, te_index in folds:
    X_train, X_test = X[tr_index], X[te_index]
    y_train, y_test = y[tr_index], y[te_index]
    bst.fit(X_train, y_train)
    y_pred = bst.predict(X_test)
    e = rmsel(y_test, y_pred)
    print(e)

    test_pred = bst.predict(test[feature_columns].values)
    xgb_pred.append(test_pred)
    es.append(e)
print(np.mean(es, axis=0))

xgb_pred = np.array(xgb_pred)
xgb_pred = np.mean(xgb_pred, axis=0)


np.percentile(xgb_pred, 0.01),np.percentile(xgb_pred, 0.015),np.percentile(xgb_pred, 5),np.percentile(xgb_pred, 20),np.percentile(xgb_pred, 30),np.percentile(xgb_pred, 62),np.percentile(xgb_pred, 70)
xgb_pred2 = xgb_pred
test['Id'] = test_df['Id']
test['merge2'] = xgb_pred2
test[['Id', 'merge2']].to_csv('result/gyq_demo.csv',index=None,header=None)
print('over')