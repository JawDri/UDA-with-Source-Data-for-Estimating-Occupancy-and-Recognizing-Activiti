VADA Method

In DIRT-T main.ipynb file run the cell for the specific task that you want to test (OE or AR, balanced or unbalanced) or wireless tasks. for AR and OE, the number of instances selected for balanced and unbalanced datasets are written in the comments.

In DIRT-T main.ipynb file choose the rate of data poison (0%, 5% or 10%).

Change the feature name in dataset.py, line 18, 19, 26,27,69,70, to 'labels' for OE and Wireless, and 'activity' for AR.

Change the feature number in dataset.py, line 53,58,85, to '9/32/912' depending of the task.

Change the n_feature in models.py, to '9/32/912' depending of the task.
Change the line 103 argument in models.py, to 'n_features * 1 * x' depending of the task (x can be 8/2/228).

TO TEST VADA: !python vada_train.py
