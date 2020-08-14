import pandas as pd
import numpy as np

ensemble_files = ["", ""]

submit = pd.read_csv("./data/submit_sample.csv", header=None)
all_prob = None
for file in ensemble_files:
    df = pd.read_csv(file,header=None)
    if all_prob == None:
        all_prob = df.iloc[:, 2]
    else:
        all_prob = all_prob + df.iloc[:, 2]

all_prob = all_prob/len(ensemble_files)

predict_label = np.argmax(all_prob, axis=1)
submit["labels"] = predict_label
if not os.path.exists("./output"):
    os.mkdir("./output")
    
submit.to_csv("./output/submit_ensemble.csv", index=False, header=False)
submit.head()