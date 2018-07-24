# HATN

Data and source code for our AAAI'18 paper "[Hierarchical Attention Transfer Network for Cross-domain Sentiment Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16873/16149)". 

# Environment
+ OS: CentOS Linux release 7.5.1804
+ GPU: NVIDIA TITAN Xp
+ CUDA: 8.0

# Requirements
+ Python 2.7.5

+ Tensorflow-gpu 1.2.1

+ numpy 1.13.3

+ nltk 3.2.1 

+ [Google Word2Vec](https://code.google.com/archive/p/word2vec/)


# Running
Before you get started, please make sure to add the following to your ~/.bashrc:

Linux:
```
export PYTHONPATH=/path/to/HATN:$PYTHONPATH
```

Centos:
```
setenv PYTHONPATH /path/to/HATN
```

#### Individual attention learning: 
The goal is to automatically capture pos/neg pivots as a bridge across domains based on PNet, which provides the inputs and labels for NPnet. If the pivots are already obtained, you can ignore this step.

```
python extract_pivots.py --train --test -s dvd [source_domain] -t electronics [target_domain] -v [verbose]
```
#### Joint attention learning
PNet and NPnet are jointly trained for cross-domain sentiment classification. When there exists large domain discrepany, it can demonstrate the efficacy of NPnet.

```
python train_hatn.py --train --test -s dvd [source_domain] -t electronics [target_domain] -v [verbose]
```
#### Training over all transfer pairs
```
./all_train.sh
```

# Citation

If the code is useful for your research, please cite our paper as follows:

```
@inproceedings{li2018hatn,
	author = {Zheng Li and Ying Wei and Yu Zhang and Qiang Yang},
	title = {Hierarchical Attention Transfer Network for Cross-Domain Sentiment Classification},
	conference = {AAAI Conference on Artificial Intelligence},
	year = {2018},
}
```
