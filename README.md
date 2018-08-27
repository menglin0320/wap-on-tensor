# wap-on-tensorflow
handwritten offline math formula recognizer

# Requirements
supported python version 3.5

Tensorflow >= 1.6.0

Open command run:
```shell
sudo sh ./installation.sh
```

Tensorflow is not on requirements.txt because if you already have gpu vesion tensorflow installed, you'll have to comment out that line. And I believe most people who's intereseted on this repor already have gpu version tensorflow installed.
You have to download data from [jianshu's repo](https://github.com/JianshuZhang/WAP).
But you have to use my scripts in folder data to generate pkls if you are using python 3.5.

# Usage
run
```shell
python3 train.py
```
to train the model.

If you want to change parameters, most of them are in the file Recognizer.py

After you train the model
run
```shell
python3 test.py <batchnumber> <set(train or valid)>
python3 test_greedy.py <batchnumber> <set(train or valid)>
```
to translate one image on a batch to latex. test.py uses beamsearch and test_greedy.py uses greedy search.

run
```shell
python3 translate_and_calc_exp.py
```
to get a text file that has all the translations on validation set. And the exp rate on validation set will be printed.

# Performance:
I didn't really tested the model's performance but LinJM told me that my model achieved exp rate 0.335 (330/985 ).
It's fairly low compare to orginal work's performance. But I used less resource.

# limitation
I didn't implement the dense encoder version of the model

I didn't implement adaptive weight noise, instead I chose to implement zoneout regularizer on lstm.
Potentially it may benefit if you apply other regularizers on vgg like encoder

# Some discussion
You may notice that for some reason I used a huge learning rate to start, but on original work the lr is set to 0.0001 at the beginning. I didn't really try to match every detail when implementing this model.
If anyone is able to figure out why I need such a huge learning rate to start, I'll be very thanksful.

There is another detail that confused me. I discussed with Jianshu about if we have to write a specific batch norm that takes the mask to make the model really work. And he told me that for some unknown reason, this model still works even if we just use normal batch norm.
But when I implement the model, if the vgg like encoder is batchnorm after relu, when I set is_training to False on batch norm, the model's performance become very poor Then I tried relu after batchnorm. It kind of works. But I don't why there's this difference or there's hidden but in my code.

# Reference:
This work is based on [Jianshu's work](https://github.com/JianshuZhang/WAP), Also you can get training and validation data on his repo.




