# HATN

Data and source code for our AAAI'18 paper "[Hierarchical Attention Transfer Network for Cross-domain Sentiment Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16873/16149)", which is an extension of our IJCAI'17 paper "[End-to-End Adversarial Memory Network for Cross-domain Sentiment Classification](https://www.ijcai.org/proceedings/2017/0311.pdf)". 

# Demo Video
Click the picture for watching a demo about visualization of attention transfer. => [![](https://github.com/hsqmlzno1/HATN/raw/master/demo.png)](https://hsqmlzno1.github.io/assets/video/hatn_visualization.mp4).

# Requirements
+ Python 2.7.5

+ Tensorflow-gpu 1.2.1

+ numpy 1.13.3

+ nltk 3.2.1 

+ [Google Word2Vec](https://code.google.com/archive/p/word2vec/)

# Environment
+ OS: CentOS Linux release 7.5.1804
+ GPU: NVIDIA TITAN Xp
+ CUDA: 8.0

# Running

### Individual attention learning: 
The goal is to automatically capture pos/neg pivots as a bridge across domains based on PNet, which provides the inputs and labels for NPnet. If the pivots are already obtained, you can ignore this step.

```
python extract_pivots.py --train --test -s dvd [source_domain] -t electronics [target_domain] -v [verbose]
```
### Joint attention learning:
PNet and NPnet are jointly trained for cross-domain sentiment classification. When there exists large domain discrepany, it can demonstrate the efficacy of NPnet.

```
python train_hatn.py --train --test -s dvd [source_domain] -t electronics [target_domain] -v [verbose]
```
### Training over all transfer pairs:
```
./all_train.sh
```

# Results

The results are obtained under ramdom seed 0 in this implementation.


| Task  | P-net  | HATN_h (full model) |
 :-: | :-: | :-:
| books-dvd           | 0.8722 | 0.8770 |
| books-electronics   | 0.8388 | 0.8620 |
| books-kitchen       | 0.8518 | 0.8708 |
| books-video         | 0.8728 | 0.8735 |
| dvd-books           | 0.8783 | 0.8802 |
| dvd-electronics     | 0.8393 | 0.8678 |
| dvd-kitchen         | 0.8467 | 0.8700 |
| dvd-video           | 0.8822 | 0.8897 |
| electronics-books   | 0.8328 | 0.8362 |
| electronics-dvd     | 0.8340 | 0.8387 |
| electronics-kitchen | 0.9010 | 0.9012 |
| electronics-video   | 0.8352 | 0.8345 |
| kitchen-books       | 0.8398 | 0.8483 |
| kitchen-dvd         | 0.8357 | 0.8473 |
| kitchen-electronics | 0.8807 | 0.8908 |
| kitchen-video       | 0.8370 | 0.8403 |
| video-books         | 0.8682 | 0.8748 |
| video-dvd           | 0.8737 | 0.8760 |
| video-electronics   | 0.8347 | 0.8585 |
| video-kitchen       | 0.8463 | 0.8602 |
| Average		         | 0.8551 | 0.8649 |


# Citation

If the data and code are useful for your research, please be kindly to give us stars and cite our paper as follows:

```
@inproceedings{li2018hatn,
	author = {Zheng Li and Ying Wei and Yu Zhang and Qiang Yang},
	title = {Hierarchical Attention Transfer Network for Cross-Domain Sentiment Classification},
	conference = {AAAI Conference on Artificial Intelligence},
	year = {2018},
}
```

```
@inproceedings{li2017end,
  title={End-to-end adversarial memory network for cross-domain sentiment classification},
  author={Li, Zheng and Zhang, Yu and Wei, Ying and Wu, Yuxiang and Yang, Qiang},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI 2017)},
  year={2017}
}
```
