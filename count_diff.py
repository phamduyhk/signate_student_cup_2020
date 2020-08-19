import pandas as pd

df1 = pd.read_csv("output/prob_bert-base-cased_128_10ep.csv",header=None)
df2 = pd.read_csv("output/prob_bert-base-cased_512_10ep.csv",header=None)

df3 = pd.read_csv("output/prob_bert-large-cased_256_10ep.csv",header=None)
df4= pd.read_csv("output/prob_albert-xxlarge-v2_512_10ep.csv",header=None)
df5 = pd.read_csv("output/prob_albert-xxlarge-v2_128_10ep.csv",header=None)

diff1 = df5.iloc[:,1] - df1.iloc[:,1]
s = 0
for item in diff1:
    s+=abs(item)
print(s)


diff2 = df4.iloc[:,1] - df3.iloc[:,1]
s = 0
for item in diff2:
    s+=abs(item)
print(s)
