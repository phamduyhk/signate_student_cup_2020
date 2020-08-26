import pandas as pd

file_name = "output/pseudo_bert-large-cased_0-192_1cv_30ep.csv"
# file_name = "submission_gru_lstm_50ep.csv"
df = pd.read_csv(file_name, header=None)

submit = pd.read_csv("./data/submit_sample.csv", header=None)
submit.iloc[:, 1] = df.iloc[:,1]
out_name = "data_augumented_pseudo_bert-large-cased_0-192_1cv_30ep.csv"
# out_name = "submit_gru_lstm_50ep.csv"
submit.to_csv(out_name, header=None, index=False)