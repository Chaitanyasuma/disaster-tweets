import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

nltk.download('stopwords')
nltk.download('wordnet')

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

# Add a length of tweet column

train_df['length'] = train_df['text'].apply(lambda x: len(x))
test_df['length'] = test_df['text'].apply(lambda x: len(x))


print(train_df.info())
print(test_df.info())

print("\nTrain Set:\n", train_df.length.describe(), '\n\nTest Set:\n', test_df.length.describe())

# Data Set of non disaster and disaster
disaster = train_df.loc[train_df['target'] == 1]
non_disaster = train_df.loc[train_df['target'] == 0]

print('Disaster mean tweet length:\n', disaster.length.mean(), 
      '\nNon-Disaster mean tweet length:\n', non_disaster.length.mean())

# What is the mean length of a true disaster tweet?
sns.barplot(x=train_df['target'], y=train_df['length'])

missing_cols = ['keyword', 'location']

fig, axes = plt.subplots(ncols=2, figsize=(15,5), dpi=100)

sns.barplot(x=train_df[missing_cols].isnull().sum().index, y=train_df[missing_cols].isnull().sum().values, 
           ax=axes[0])
sns.barplot(x=test_df[missing_cols].isnull().sum().index, y=test_df[missing_cols].isnull().sum().values, 
           ax=axes[1])

axes[0].set_ylabel('Missing values')

axes[0].set_title('Training Set')
axes[1].set_title('Test Set')

print(train_df[missing_cols].nunique())

# Group the rows by the 'keyword' and average their target column together
train_df['mean_keyword'] = train_df.groupby('keyword')['target'].transform('mean')

#print(train_df['mean_keyword'][0:50])

fig = plt.figure(figsize=(8,72), dpi=100)

# Then count each keyword and graph
sns.countplot(y=train_df.sort_values(by='mean_keyword', ascending=False)['keyword'],
              hue=train_df.sort_values(by='mean_keyword', ascending=False)['target'])

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc=1)
plt.title('Target Distribution in Keywords')

#plt.show()

train_df.drop(columns=['mean_keyword'], inplace=True)

disaster['CAPS_count'] = disaster['text'].str.count(r'[A-Z]')

non_disaster['CAPS_count'] = non_disaster['text'].str.count(r'[A-Z]')

sns.barplot(x = ['Disaster', 'Non Disaster'], y = [disaster['CAPS_count'].mean(), non_disaster['CAPS_count'].mean()])

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

#train_df['text'] = train_df['text'].apply(lambda x: remove_emoji(x))


vectorizer = TfidfVectorizer(ngram_range=(1,3),max_features=10000)
X = vectorizer.fit_transform(train_df['text'])
y = train_df['target']

gbc = GradientBoostingClassifier(random_state=42)
params_grid = {
    "n_estimators":[100,200,300,400,500],
    "max_depth":[3,5,10]
}

f1 = make_scorer(f1_score , average='macro')

gcv = GridSearchCV(estimator = gbc, 
             param_grid = params_grid, 
             cv = 3, 
             n_jobs = -1, 
             verbose = 4,
             scoring=f1)

gcv.fit(X,y)

print("Best F1 score : ", gcv.best_score_)

# Logistic Regression Model
#log_reg = LogisticRegression(random_state=42, solver='saga', max_iter=5000)
log_reg = LogisticRegression(random_state=0, solver='saga', max_iter=5000)

params_grid_lr = {
    "penalty": ["l1", "l2"],               
    "C": [0.001, 0.01, 0.1, 1, 10, 100],  
    "class_weight": [None, "balanced"]
}

gcv_lr = GridSearchCV(estimator=log_reg, 
                      param_grid=params_grid_lr, 
                      cv=3, 
                      n_jobs=-1, 
                      verbose=1,
                      scoring=f1)
gcv_lr.fit(X, y)

print("Best F1 score (Logistic Regression): ", gcv_lr.best_score_)