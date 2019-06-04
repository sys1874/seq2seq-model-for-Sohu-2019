# Sohu 2019 内容识别大赛 
## 基于seq2seq的端到端核心实体识别和情感预测（初赛rank11-sys1874）




## 环境要求
* python>=3.6
* pytorch>=0.4.1
* tqdm
* [elmo(可选)](https://github.com/HIT-SCIR/ELMoForManyLangs)

## 模型设计
![](/extend/model_struct.png)
### 示例：
文章：装机容易忽视的**组件**，主机**电源**不能随便选对于大部分入门玩家来说，在电源的选择上往往都比较随意。
1. 按照核心实体在原文的位置，构造解码序列。如上述例子，解码序列为： （1）**组件**、 （2）**电源**
2. 利用seq2seq 的attention机制去标注实体。训练时，使得原文中目标实体对应的attention score最大；预测时，attention score最大的单词为目标实体。
3. 利用seq2seq 的输出层预测实体对应的情感。
4. 利用beam search，使得预测出的 核心实体训练总体得分最大化。

### 实验结果(初榜测试集)：

|Model | 描述  | 实体得分 | 情感得分 | 总分 |
| :---------------: |:---------------:| :---------------:| :---------------:| :---------------:|
| baseline      | seq2seq+beam_search+glove | 0.5468 | 0.2885 | 0.4177 |
| ensemble      | 集成GRU/LSTM        | 0.5623 | 0.3392 | 0.4508 |
| Baseline+elmo | 添加elmo       |   0.5660 | 0.3369 | 0.4515|
|ensemble(elmo)|集成添加了elmo的LSTM、GRU|0.5783|0.3525|0.4654|
|ensemble(elmo)+正向最大匹配|利用正向最大匹配增加候选实体|0.5857|0.3594|0.4725|





## Usage

### Step 1: 执行preprocess.ipynb进行数据预处理
处理完的结果如下

```
./data/model.train
./data/model.test.raw
./data/entity.txt
```

### Step 2: 训练 Model

训练baseline

```
python network.py --data_dir ./data/  --data_prefix model  --save_dir ./models/baseline    --gpu 0     --max_vocab_size 50000   --min_freq  5 --entity_file ./data/entity.txt   \
--lr 0.0003  --hidden_size 1024  --num_layers 2   --attn mlp  --log_steps 100  --valid_steps 300  --batch_size 64   --pretrain_epoch -1     --lr_decay 0.5
```
训练baseline+elmo

```
python network.py --data_dir ./data/  --data_prefix model  --save_dir ./models/baseline_elmo    --gpu 0     --max_vocab_size 50000   --min_freq  5 --entity_file ./data/entity.txt   \
--lr 0.0003  --hidden_size 1024  --num_layers 2   --attn mlp   --log_steps 100    --valid_steps 1300 --batch_size 15    --pretrain_epoch -1     --lr_decay 0.5  --elmo
```
还有一些其他的参数，如更换RNN类型 --rnn_type, 使用词性标注 --POS ，大家可以阅读network.py，自行尝试。


### Step 3: 测试 Model

```
python network.py --data_dir ./data/  --data_prefix model  --save_dir ./models/baseline    --gpu 0     --max_vocab_size 50000   --min_freq  5 --entity_file ./data/entity.txt   \
--lr 0.0003  --hidden_size 1024  --num_layers 2   --attn mlp  --log_steps 100  --valid_steps 300  --batch_size 64   --pretrain_epoch -1     --lr_decay 0.5 \ 
--ckpt  ./emo_models/layer_2_64_embed_19/best.model   --test     --beam_size 5  --gen_file   ./result/new_result_emo12.txt   --for_test
```
ensemble.ipynb baseline model 平均融合 \
ensemble_elmo.ipynb elmo model 平均融合

## acknowledge
该项目代码是在 百度2019-知识驱动的多轮对话的[baseline](https://github.com/baidu/knowledge-driven-dialogue/tree/master/generative_pt)上进行改进的。\
同时，我们在[百度2019-知识驱动的多轮对话](http://lic2019.ccf.org.cn/talk)中取得**自动评测Top1，人工评测Top2**的成绩，相关代码将在后续进行开源。

