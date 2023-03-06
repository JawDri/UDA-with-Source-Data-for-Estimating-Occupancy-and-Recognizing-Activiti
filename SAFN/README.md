SAFN Method

In SAFN.ipynb file run the cell for the specific task that you want to test (OE or AR, balanced or unbalanced). for AR, the number of instances selected for balanced and unbalanced datasets are written in the comments.

In SAFN.ipynb file choose the rate of data poison (0%, 5% or 10%).

Change the number of features in net.py file (len(FEATURES) = 9 for OE and len(FEATURES), line 16.

Change the task in train.py, train_ent.py and eval.py (class_num) depending of the task that you want to test, example: ##number of classes##=3 in case of task with 3 labels.

TO TRAIN SAFN: !python train.py 
TO TRAIN SAFN+ENTM: !python train_ent.py 
TO TEST: !python eval.py
