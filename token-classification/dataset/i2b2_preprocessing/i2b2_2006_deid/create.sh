python to_conll.py deid_surrogate_train_all_version2.xml > train_dev.conll
python split_train_dev.py
python to_conll.py deid_surrogate_test_all_groundtruth_version2.xml > ../../NER/2006/test.txt
python get_labels.py
mv train.conll ../../NER/2006/train.txt
mv dev.conll  ../../NER/2006/dev.txt
