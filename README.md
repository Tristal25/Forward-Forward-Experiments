# Forward-Forward-Experiments

This project aims to investigate and demystify the 
Forward-Forward Algorithm in Hinton’s (2022) 
recent paper. A novel procedure of initializing 
negative data is introduced, which includes random 
initialization within each epoch. This enables the 
algorithm to cover more negative cases within limited 
speed decrease. Logistic loss function with a new parameter
called margin is proposed  in this project, which improves
the training result under same model configurations. 
Moreover, we clarified the method used in the algorithm 
and tested it under different model sizes, activation 
functions, and other related hyperparameters.

## Repository Roadmap

- [Report PDF File](report.pdf)
- [Main Script](main.py)
- [Model Classes and Unility Functions](util/)
- [Model Arguments and Configurations](argdata.py)
- [Data](data/)
- [Images for Results](images/)

## Usage

Change arguments in `argdata.py` and run the following command:

```bash
python main.py
```

## Results

We run the algorithm using this configuration:

```python
argDict = {
        'device': 'gpu',
        'dataset': 'mnist',
        'threshold': 10,
        'lr': 0.03,
        'epochs': 2000,
        'train_batch_size': 10000,
        'test_batch_size': 10000,
        'num_layers': 4,
        'hidden_size': 500,
        'random_seed': 42,
        'norm': 1,
        'dropout': 0.2,
        'skip_connection': 0,
        'unsupervised': 1,
        'activation': 'elu',
        'margin': 1,
        'loss': 'logistic',
        'neg_data': 'random'
    }
```

The resulting test accuracy is 97.36%. 

