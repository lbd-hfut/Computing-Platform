config = {
    "print_freq": 100,
    "checkpoint_path": "./weights/checkpoint/",
    "model_path": "./weights/models/",
    "data_path": './data/train/',
    "warm_lr": 0.001,
    "train_lr": 0.0005,
    "max_iter": 20,
    "weight_decay": 3e-2,
    "layers": [2, 50, 50, 50, 2],
    "scale": [[1,1]]*5+[[10,10]]*5,
    "warm_adam_epoch": 10,
    "warm_bfgs_epoch": 21,
    "train_adam_epoch": 10,
    "train_bfgs_epoch": 21,
    "patience_adam": 10,
    "patience_lbfgs": 10,
    "delta_warm_adam": 0.01,
    "delta_warm_lbfgs": 0.005,
    "delta_train_adam": 0.003,
    "delta_train_lbfgs": 0.001,
    "epoch": 0,
    "print_feq": 10
}
