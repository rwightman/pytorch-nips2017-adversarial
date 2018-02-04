# NIPS 2017 Adversarial Competition (PyTorch) 

This repository contains the code that Aleksey (https://github.com/alekseynp) and I wrote for the NIPS 2017 Adversarial challenges:

* Non-Targeted Attack: https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack
* Targeted Attack: https://www.kaggle.com/c/nips-2017-targeted-adversarial-attack
* Defense: https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack

The code in this repository includes the competition runtime scripts, along with support code for experiments, training, and other ideas at various stages of completion. The code here is far from production quality, and many pieces were abandoned moving from one idea to the next.

The runtime code submitted for the final competition round can be found at: https://bitbucket.org/alekseynp/nips-submissions/src 

Additionally, Aleksey wrote a paper to describe our work and give credit where credit is due (http://alekseynp.com/papers/nips2017-adversarial-paper.pdf) and we did a presentation at our local (Vancouver, Canada) Kaggle meetup with some additional information (https://goo.gl/du57Zk)

Of potential interest to anyone rooting through this code:

1. There is work done by Aleksey experimenting with the Madry challenge on the anp-madry branch that has not been merged to Master.

1. The `train_adversarial_defense.py` script contains a PyTorch implementation of what I call 'Ensemble-Ensemble Adversarial Training'. Inspired by the ideas in https://arxiv.org/abs/1705.07204, it trains a weighted ensemble of base defense networks against an ensemble of different attacks (themselves optimizing perturbations against ensembles of base networks). It is resource heavy and was tested on a 4 x P100 GCP instance. It truly reflects the flexibility of PyTorch as an experimentation platform. 

1. I ported weights of the Google provided adversarially trained Inception-V3 and ensemble adversarially trained Inception-Resnet-V2 to PyTorch models. Anyone interested in those weights or the porting code can contact us as we don't have suitable hosting.
