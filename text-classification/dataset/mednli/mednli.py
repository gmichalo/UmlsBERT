import pandas as pd


class mednil:
    def __init__(self, path1, path_med):
        self.path1 = path1
        self.path_med = path_med
        self.label = {}
        self.label[1] = 'entailment'
        self.label[0] = 'contradiction'
        self.label[2] = 'neutral'


    def read(self, path2, path3):
        list_dataset = []
        dataset = self.path_med + path2
        f = open(dataset, "r")
        for x in f:
            sentence1 = x.split('sentence1":')[1].split("pairID")[0][2:-4].strip()
            sentence2 = x.split('sentence2":')[1].split("sentence2_parse")[0][2:-4].strip()
            label = x.split("gold_label")[1][4:-3].strip()
            list_dataset.append([sentence1, sentence2, label])
        df = pd.DataFrame(list_dataset, columns=["sentence1", "sentence2", "label"])


        df.to_csv(self.path1 + path3, index=False, sep="\t")
        return


reader = mednil("mednli/", 'mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0/')
reader.read("mli_train_v1.jsonl", "train.tsv")
reader.read("mli_dev_v1.jsonl", "dev_matched.tsv")
reader.read("mli_test_v1.jsonl", "test_matched.tsv")
