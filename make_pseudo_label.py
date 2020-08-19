import pandas as pd

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")


trained_output_file = "./output/prob_albert-xxlarge-v2_256_10cv_5ep.csv"
trained_output = pd.read_csv(trained_output_file, header=None)
pseudo_label = trained_output.iloc[:,1]
test["jobflag"] = pseudo_label

submit = pd.read_csv("./data/submit_sample.csv", header=None)
all_prob = None

for i, item in enumerate(trained_output.iloc[:,2]):
    arr = eval(item)
    for v in arr:
        if v>0.5:
            row = test.loc[i]
            row["jobflag"] = trained_output.iloc[i,1]
            # print(row)
            train.loc[len(train)+1]= row

print(train)
train.to_csv("data/pseudo_train.csv", index=False)