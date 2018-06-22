cd data
tar -xvzf off_image_test.tar.gz
tar -xvzf off_image_train.tar.gz
python3 gen_pkl_train.py
python3 gen_pkl_test.py
