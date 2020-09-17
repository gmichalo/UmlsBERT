file = open('train_dev.conll', 'r')
Lines = file.readlines()
labels = {}
count = 0
for line in Lines:
    try:
        label = line.split(" ")[1].replace("\n","")
    except:
        pass
    if label not in labels:
        labels[label] = 0

f = open("../../NER/2006/label.txt", "w")
for label in reversed(sorted(labels)):
    f.write(label+"\n")
f.close()