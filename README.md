# Horse or Deer / 馬鹿判定器
This system can distinguish "Horse" from "Deer."

![Horse or Deer](https://user-images.githubusercontent.com/53997809/64553273-0a100580-d374-11e9-9bdc-e3c5a0e6f736.png)

## Demo Site

https://horse-deer.herokuapp.com/

## Installation

* python 3.6.8
* flask 1.0.3
* opencv-python 3.4.5.20
* requests 2.22.0
* tensorflow 1.13.1

Install the following packages:

```
$ pip install flask
$ pip install opencv-python
$ pip install requests
$ pip install tensorflow
```

Clone this repository:

```
$ git clone https://github.com/yoshy-toshy/horse-deer.git
```

Start the server:

```
$ cd horse-deer
$ python web.py
```

Access http://localhost:5000/, and you can see the top page.

## Usage

Click on the button "画像を選択する" to select an image, and then push the judge button "判定."

If the system identifies the subject of the selected image as horse, it will show "馬." For deer, it will show "鹿."


## How to Make a New AI Model

You can make your own AI model which can distinguish several image classes instead of "horse vs. deer." (This repository does not have data files because they are git-ignored. To train the AI, you need to collect data by yourself.)

### 1. Plan

Decide what kinds of image your AI supports.

e.g. "Cabbage vs. Lettuce," "iPhone vs. Android," etc.

### 2. Create Folders

Collect many image files for training data and validation data. The number of Class0 data must be the same as that of Class1.

Put them into the folders `./data/train/0/`, `./data/train/1/`, `./data/validate/0/`, and `./data/validate/1/`.

For example, you can put the files as below:

|Folder |File name | Number of files |
|----|----|----|
|./data/train/0/ |horse001.jpg<br>horse002.jpg<br>horse003.jpg<br>...|250 |
|./data/train/1/ |deer001.jpg<br>deer002.jpg<br>deer003.jpg<br>...|250 |
|./data/validate/0/ |horse351.jpg<br>horse352.jpg<br>horse353.jpg<br>...|50 |
|./data/validate/1/ |deer351.jpg<br>deer352.jpg<br>deer353.jpg<br>...|50 |


After collecting the data, execute the following file to update the definition files `./data/train/data.txt` and `./data/validate/data.txt`:

```
$ python update_data.py
```


### 3. Modify Hyperparameters

You can modify the following hyper-parameters in `main.py`:
* IMAGE_SIZE
* PATTERN_SIZE
* MAX_STEPS
* BATCH_SIZE
* LEARNING_RATE
* NODES
* KEEP_PROB

Basically, you can use the above constants without modification, but please make sure that BATCH_SIZE must divide the number of your training data evenly.

### 4. Rename Class Labels

Rename class labels in `main.py`:

* LABELS

If your target images are "iPhone" and "Android," change "馬" and "鹿" to "iPhone" and "Android."

### 5. Remove the Existing Model

Remove the existing AI model files:

```
$ rm ./model/*
```

After this operation, you can no longer use the default model.

### 6. Train Your AI Model

Execute `main.py` to start machine learning:

```
$ python main.py
```

### 7. Use Your AI

Start the server after `main.py` finishes successfully:

```
$ python web.py
```

Access http://localhost:5000/, and enjoy your AI.

## Disclaimer
* I do not take responsibility for any damage caused by this system.
