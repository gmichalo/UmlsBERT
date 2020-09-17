import pandas as pd
from tqdm import tqdm
import re

import nltk

class create_mimic_embedding:
    def __init__(self, file_name):
        self.file_name = file_name

    def read_notes(self):
        notes_list = []
        df_notes = pd.read_csv(self.file_name)
        a_file = open("mimic_string.txt", "w+")
        for row in tqdm(df_notes.iterrows()):
            text = row[1]["TEXT"]
            text = text.replace("[**", " ")
            text = text.replace("**]", " ")
            text = text.replace("\n", " ")
            text = re.sub(' +', ' ', text)
            sent_text = nltk.sent_tokenize(text)
            for sentence in sent_text:
                if len(sentence.split(" "))> 3:
                    a_file.write(sentence+"\n")
 
        return




object1 = create_mimic_embedding("NOTEEVENTS.csv")
object1.read_notes()
