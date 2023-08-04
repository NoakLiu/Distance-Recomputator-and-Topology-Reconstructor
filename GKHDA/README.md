Excute the following instruction
```
nohup python -u training.py --dataset cora > GKHDA_cora.txt
```

A more self-defined linux command line example can be executed below:

```
nohup python -u training.py --fastmode store_true --epochs 1000 --patience 100 --hidden 8 --dropout 0.5=6 --nb-heads 8 --alpha 0.2 --lr 0.005 --weight-decay 5e-4 --k 5 --beta 0.1 --num-sample 10 --dataset pubmed
```