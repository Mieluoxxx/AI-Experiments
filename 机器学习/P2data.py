import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/P2/adult.data', sep=",", header=None)
# Attribute names
attributes = [
    "年龄", "工作类别", "人口普查员序号", "教育程度", "教育年数",
    "婚姻状况", "职业", "家庭关系", "种族", "性别",
    "资本收益", "资本支出", "每周工作小时数", "原籍国", "收入"
]

# Using pandas to read the data
df.columns = attributes

df.replace("?", pd.NaT, inplace=True)

trans = {'工作类别':df['工作类别'].mode()[0],
         '职业':df['职业'].mode()[0],
         '原籍国':df['原籍国'].mode()[0],
         }
df.fillna(trans, inplace=True)


X = df.drop('收入', axis=1)
X = X.map(lambda x: x.strip() if type(x) is str else x)
y = df['收入'].str.strip()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
# 将X_train和y_train合并为一个DataFrame
train = pd.concat([X_train, y_train], axis=1)
# 将X_test和y_test合并为一个DataFrame
# 保存train.csv和test.csv
train.to_csv('data/P2/train.csv', index=False)
X_test.to_csv('data/P2/test.csv', index=False)
y_test.to_csv('data/P2/true.csv', index=False)