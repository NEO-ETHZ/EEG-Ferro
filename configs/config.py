import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Transfer learning for SNNs")

    # Dataset
    parser.add_argument("--dataset", default="EEGDataset2DLeftRight", type=str, help="Dataset name")

    # Seed
    parser.add_argument("--seed", type=int, default=5, help="Random seed")



    # Unfreeze flag
    parser.add_argument("--unfreeze_layers", nargs="+", default=["conv1", "conv2", "conv3", "tc1", "r1", "fc1", "fc2"],
                    help="List of layers to keep trainable (e.g. --unfreeze_layers fc1 fc2)")
    
    # Freeze flag for parameters
    parser.add_argument("--freeze_parameters", nargs="+", default=[], help="List of parameters to freeze")
    
    # Reinitialize layer
    parser.add_argument("--reinit_layers", nargs="+", default=[])

    # RNN layer density
    parser.add_argument("--RNN_density", type=float, default=1.0)

    # Learning rates
    # Finetune Learning rates
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="pretraining learning rate for synaptic weights")
    
    # Finetune Learning rates
    parser.add_argument("--finetune_lr", type=float, default=2e-4,
                        help="finetune learning rate for synaptic weights")
  

    parser.add_argument("--wt_decay", type=float, default=1e-6)
    parser.add_argument("--ts_decay", type=float, default=2e-6)
    parser.add_argument("--neuron_decay", type=float, default=2e-6)

    parser.add_argument("--validation_ratio", type=float, default=0.5)


    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--pretraining_batch_size", type=int, default=64)

    parser.add_argument("--retuning_batch_size", type=int, default=64)
    parser.add_argument("--finetuning_batch_size", type=int, default=1)
    parser.add_argument("--pretrain_epochs", type=int, default=20)
    parser.add_argument("--retuning_epochs", type=int, default=4)
    parser.add_argument("--finetune_epochs", type=int, default=5)
    parser.add_argument("--thr", type=float, default=0.025)
    parser.add_argument("--thr_asym", type=float, default=1.0)
    parser.add_argument("--x", type=float, default=4.0) # Noise 1/RSD

    parser.add_argument("--dut", type=str, default="dut2")
    parser.add_argument("--A_scale", type=float, default=1.0)
    parser.add_argument("--A_asym", type=float, default=1.0)
    parser.add_argument("--train_decays", type=str, default="False") # Whether to train decay parameters during re-tuning
    parser.add_argument("--rsd_c2c", type=float, default=0.0)
    parser.add_argument("--sigma_add", type=float, default=0.0)
    parser.add_argument("--sigma_mul", type=float, default=0.0)

    parser.add_argument("--nb_folds", type=int, default=5)
    parser.add_argument("--q", type=int, default=0)

    return parser.parse_args()