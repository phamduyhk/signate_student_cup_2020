import pandas as pd

file_name = "output/prob_bert-large-cased_256_10ep.csv"
df = pd.read_csv(file_name, header=None)

submit = pd.read_csv("./data/submit_sample.csv", header=None)
submit.iloc[:, 1] = df.iloc[:,1]
submit.to_csv("output/bert-large-cased_256_10ep.csv", header=None, index=False)