To run the file, execuate the following instruction:
```
nohup python -u GDRA.py --dataset cora > cora_basic_cul.txt
```

A more self-defined linux command line example can be executed below:

```
nohup python -u GDRA.py --fastmode store_true --epochs 1000 --patience 100 --hidden 8 --dropout 0.5=6 --nb-heads 8 --alpha 0.2 --lr 0.005 --beta 0.1 --num-sample 10 --dataset pubmed
```

