This project trains a deep reinforcement learning-based hierarchical text classification model. Find details in the paper.

How to run the codes:

Download yelp's 2018 dataset from https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset

Unzip the file and save "yelp_academic_dataset_business.json" and "yelp_academic_dataset_review.json" in the "dataset" folder, which also contains the hierarchy file.

Run

'''
pip install -r requirements.txt 
'''

Then run 

python make_dataset.py

To generate the preprocessed data, which will be saved as "train_data.pickle" in the "dataset" folder.

Then run 

python main.py

to start training. The model will be saved as "model.pt" in this folder. Feel free to change the hyperparameters in "config.py".

A demo of the label assignment strategy at different time steps, named "demo.jpg", is also included in the folder.