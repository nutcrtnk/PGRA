# PGRA

The code for the paper "[PGRA: Projected Graph Relation-Feature Attention Network for Heterogeneous Information Network Embedding.](https://doi.org/10.1016/j.ins.2021.04.070)"

## Requirements

We tested the code on:
* python 3.6
* pytorch 1.5.1
* networkx 2.3

other requirements:
* numpy
* pandas

## Usage

Run the code using the command:
```
python src/main.py [Dataset_Name] [options]
```

## Options
For PGRA-DistMult, use default hyperparameters

For PGRA-TransH, use
```
--score [l1/l2] --pre transh
```

For the best neighbor regularization settings (lambda) on DBLP/Yelp/DM/Aminer, use
```
--best_lambda
```
or to manually set, use
```
--nb_reg [value]
```