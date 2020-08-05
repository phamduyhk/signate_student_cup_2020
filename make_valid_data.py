import pandas as pd
file_path = "./data/train.csv"

df = pd.read_csv(file_path)
df = df.drop("id", axis=1)
df['jobflag'] = df['jobflag'] - 1
length = len(df)
start_point = int(length*2/3)
train = df[:start_point]
valid = df[start_point:]
train.to_csv("./data/train_data.csv", index=False)
valid.to_csv("./data/valid_data.csv", index=False)

test = pd.read_csv("./data/test.csv")
test = test.drop("id", axis=1)
test['jobflag'] = [0 for i in range(len(test))]
test.to_csv("./data/test_data.csv", index=False)