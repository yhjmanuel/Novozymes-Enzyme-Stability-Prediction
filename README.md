Ranking: 14 / 2483

Please find the training data that we used in this project at [https://drive.google.com/drive/folders/1OzALs-Zlo_IEVbTMr0e7MswW83VJCX6q?usp=share_link]

Dependencies are specified in "requirements.txt"

Scripts:
"cnn_train.py": How we used CNN-based methods to train a model 
"pbert_cnn_train.py": How we used ProteinBERT and CNN to train a model. Our final solution.
"pbert_cnn_inference.py": After training a ProteinBERT+CNN-based model, we used this script to do the inference.
(In our version, the last two scripts use both device 'cpu' and device 'mps'. You can change the 'mps' device to 'cuda' in the script. This solution needs GPU) 

After downloading training data, run

python cnn_train.py 

to train a CNN-based model. Also, run

python pbert_cnn_train.py

to train a ProteinBERT+CNN-based model.

Feel free to change training configs in each training script's "TrainConfig" class. 

Thank you for reading
