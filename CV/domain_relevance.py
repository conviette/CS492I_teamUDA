from baseline_data import pseudo_data_names, pseudo_label, pseudo_probs
import json


dataset = dict()
for i in range(len(pseudo_label)):
    if pseudo_label[i] in dataset:
        dataset[pseudo_label[i]].append((pseudo_data_names[i], pseudo_probs[i]))
    else:
        dataset[pseudo_label[i]]=[(pseudo_data_names[i], pseudo_probs[i])]
files = []
for k in dataset:
    dataset[k].sort(key=lambda x:x[1], reverse=True)
    files.extend(dataset[k][:len(dataset[k])*2//3])

files = list(map(lambda x:x[0], files))

with open('domain_rel.json', 'w') as f:
    json.dump(files, f)
