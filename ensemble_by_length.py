import pandas as pd

files = ["output/prob_bert-large-cased_0-128_5cv_10ep.csv", "output/prob_bert-large-cased_128-256_5cv_10ep.csv"]

submit = pd.read_csv("./data/submit_sample.csv", header=None)

prob128 = pd.read_csv(files[0], header=None)
prob256 = pd.read_csv(files[1], header=None)

for i, row in enumerate(submit):
    length128 = prob128.iat[i,2]
    if length128 == 1:
        submit.iat[i,1] = prob128.iat[i,1]
    else:
        submit.iat[i,1] = prob256.iat[i,1]

print(submit) 
submit.to_csv("submit_ensemble_by_length.csv",header=None,index=False)
