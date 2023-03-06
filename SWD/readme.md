SWD Method

In SWD.ipynb file run the cell for the specific task that you want to test (OE or AR, balanced or unbalanced). for AR, the number of instances selected for balanced and unbalanced datasets are written in the comments. 

In SWD.ipynb file choose the rate of data poison (0%, 5% or 10%).


Change the task in swd.py (number of cells in the two classifiers in toyNet function) depending of the task that you want to test, example: class_num=3 in case of task with 3 labels.


TO TEST SWD: !python swd.py -mode adapt_swd
