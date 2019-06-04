# Sohu 2019 内容识别大赛 
## 基于seq2seq的端到端核心实体识别和情感预测（初赛rank11）




## Requirements

* cuda=9.0
* cudnn=7.0
* python>=3.6
* pytorch>=1.0
* tqdm
* numpy
* nltk
* scikit-learn

## Quickstart

### Step 1: Preprocess the data

Put the data provided by the organizer under the data folder and rename them  train/dev/test.txt: 

```
./data/resource/train.txt
./data/resource/dev.txt
./data/resource/test.txt
```

### Step 2: Train the model

Train model with the following commands.

```bash
sh run_train.sh
```

### Step 3: Test the Model

Test model with the following commands.

```bash
sh run_test.sh
```

### Note !!!

* The script run_train.sh/run_test.sh shows all the processes including data processing and model training/testing. Be sure to read it carefully and follow it.
* The files in ./data and ./model is just empty file to show the structure of the document.
