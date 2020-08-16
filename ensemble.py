import pandas as pd
import numpy as np
import os

ensemble_files = ["./output/prob_albert-xxlarge-v2_128_10ep.csv", "output/prob_bert-large-cased_256_10ep.csv"]

submit = pd.read_csv("./data/submit_sample.csv", header=None)
all_prob = None
for file in ensemble_files:
    df = pd.read_csv(file,header=None)
    array = np.zeros(shape=(len(df),4))
    for i, item in enumerate(df.iloc[:, 2]):
        arr = eval(item)
        array[i] = arr

    if all_prob is None:
        all_prob = array
    else:
        all_prob = all_prob + array

# print(all_prob)

print(all_prob)
predict_label = np.argmax(all_prob, axis=1)
predict_label +=1
print(predict_label)
submit.iloc[:,1] = predict_label
if not os.path.exists("./output"):
    os.mkdir("./output")
    
submit.to_csv("./output/submit_ensemble_{}_{}.csv".format("bert-large-cased_256", "prob_albert-xxlarge-v2_128"), index=False, header=False)
submit.head()