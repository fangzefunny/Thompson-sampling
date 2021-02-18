# Thompson-sampling

This is a repo that replicates the Thompson sampling (Russo et al, 2017) and rate-distortion (Arumugam, 2021).
For purpose of getting the intuition of how these algorithms work, this repo only focuses on simple Bernoulli bandit conditon.

## Where to start 

To compare Thompson sampling, e-greedy, and UCB algorithm, run:

        python Russo17.py
        
To compare Thompson sampling and rate-distortion based Thompson sampling, run:
        
        python Arumugam21.py
        
If you want to change the experimental set, find the hyperparam function and tune the parameter.

## Reference

Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2017). A tutorial on thompson sampling. arXiv preprint arXiv:1707.02038.
Chicago	

Arumugam, D., & Van Roy, B. (2021). Deciding What to Learn: A Rate-Distortion Approach. arXiv preprint arXiv:2101.06197.

Thanks to: https://github.com/andrecianflone/thompson
