config = {
    "checkpoint_path": "./weights/checkpoint/",
    "model_path": "./weights/models/",
    "data_path": './data/train2/',
    "warm_lr": 0.01,
    "train_lr": 0.001,
    "max_iter": 10,
    "weight_decay": 1e-6,
    "layers": [2, 50, 50, 50, 1],
    "warm_adam_epoch": 100,
    "warm_bfgs_epoch": 100,
    "train_adam_epoch": 100,
    "train_bfgs_epoch": 100,
    "patience_adam": 30,
    "patience_lbfgs": 30,
    "delta_warm_adam": 0.5,
    "delta_warm_lbfgs": 0.1,
    "delta_train_adam": 0.05,
    "delta_train_lbfgs": 0.01,
    "epoch": 0,
    "print_feq": 10,
    "Batchframes": 3
}
