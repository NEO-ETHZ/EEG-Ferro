import os
import sys
import time
import threading
import psutil
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    _REPO_ROOT / "dataset",
    _REPO_ROOT / "configs",
    _REPO_ROOT / "src" / "models",
    _REPO_ROOT / "src" / "devices",
):
    sys.path.insert(0, str(_path))

import torch
import numpy as np
import copy
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
from dataset import ToTensor, EEGDataset2DLeftRight
from utility import samples_per_class, train_validate_split_5_fold_CV, train_validate_split_pretraining
from torch.utils.data import Dataset, DataLoader, sampler
import torch.optim as optim
import h5py
from datetime import datetime
from snn import WrapCUBASpikingCNN
from config import parse_args
import random
from quantization import quantize_all_weights, add_gaussian_noise_levelwise
import glob



class PerformanceMonitor:
    """Monitor training performance metrics for CPU/GPU comparison"""
    
    def __init__(self, device):
        self.device = str(device)
        self.start_time = None
        self.epoch_times = []
        self.batch_times = []
        self.forward_times = []
        self.backward_times = []
        self.optimizer_times = []
        self.data_loading_times = []
        
        # System monitoring
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.monitoring = False
        self.monitor_thread = None
        
        # Throughput metrics
        self.samples_processed = 0
        self.batches_processed = 0
        
        print(f"Performance Monitor initialized for device: {self.device}")
    
    def start_monitoring(self):
        """Start system resource monitoring in background thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_resources(self):
        """Background thread function to monitor system resources"""
        while self.monitoring:
            # CPU and RAM usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory_info.percent)
            
            # GPU memory (if available)
            if 'cuda' in self.device.lower():
                try:
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                        self.gpu_memory_usage.append(gpu_memory)
                except:
                    pass
            
            time.sleep(1.0)  # Monitor every second
    
    def start_training(self):
        """Mark start of training"""
        self.start_time = time.time()
        self.start_monitoring()
        print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def start_epoch(self):
        """Mark start of epoch"""
        return time.time()
    
    def end_epoch(self, epoch_start_time, epoch_num, train_loss, val_acc):
        """Record epoch completion"""
        epoch_time = time.time() - epoch_start_time
        self.epoch_times.append(epoch_time)

def test_accuracy(network, test_loader, device):
    """
    Return the accuracy of the prediction of the network compared to the ground truth of test data.
    :param network: Trained Pytorch network
    :param test_loader: Dataloader for test data
    :param device: device
    :return: overall accuracy, class accuracy
    """
    with torch.no_grad():
        class_correct = np.zeros(2)
        class_total = np.zeros(2)
        for data in test_loader:
            eeg_data, label = data
            eeg_data = eeg_data.to(device)
            output = network(eeg_data)
            _, predicted = torch.max(output, 1)
            c = (predicted.to('cpu') == label).numpy()
            
            for i in range(label.size(0)):
                la = label[i]
                class_correct[la] += c[i]
                class_total[la] += 1.

    return class_correct.sum()/class_total.sum(), class_correct / class_total

def save_parameters_to_h5(net, filepath, epoch=None, stage="", save_interval=5, best_epoch = False):
    """
    Save all weight matrices and trainable neuron parameters to H5 file
    :param net: neural network
    :param filepath: path to save H5 file
    :param epoch: current epoch number (None for beginning/end)
    :param stage: stage of training ("begin", "end", or "epoch")
    :param save_interval: save parameters every N epochs (default: 5)
    """
    # Skip saving if it's an epoch stage but not at the save interval
    if stage == "epoch" and epoch is not None and (epoch % save_interval != 0) and not best_epoch:
        return
    elif stage == "pretrain" and epoch is not None and (epoch % save_interval != 0) and not best_epoch:
        return
    elif stage == "finetune" and epoch is not None and (epoch % save_interval != 0) and not best_epoch:
        return
    
    with h5py.File(filepath, 'a') as f:  # 'a' for append mode
        # Create group name based on stage and epoch
        if stage == "begin":
            group_name = "initial_parameters"
        elif stage == "end":
            group_name = "final_parameters"
        elif stage == "pretrain":
            group_name = f"pretrain_epoch_{epoch:03d}" if epoch is not None else "pretrain"
        elif stage == "finetune":
            group_name = f"finetune_epoch_{epoch:03d}" if epoch is not None else "finetune"
        else:  # epoch
            group_name = f"epoch_{epoch:03d}"
        
        # Create or get the group
        if group_name in f:
            del f[group_name]  # Remove if exists to overwrite
        group = f.create_group(group_name)
        
        # Save timestamp
        group.attrs['timestamp'] = datetime.now().isoformat()
        group.attrs['stage'] = stage
        if epoch is not None:
            group.attrs['epoch'] = epoch
        
        # Iterate through all named parameters and save them
        for name, param in net.named_parameters():
            if param.requires_grad:  # Only save trainable parameters
                # Replace dots with underscores for HDF5 compatibility
                param_name = name.replace('.', '_')
                # Convert to numpy and save
                param_data = param.detach().cpu().numpy()
                group.create_dataset(param_name, data=param_data, compression='gzip')
                # Store parameter metadata
                group[param_name].attrs['original_name'] = name
                group[param_name].attrs['shape'] = param_data.shape
                group[param_name].attrs['dtype'] = str(param_data.dtype)

def list_parameters(net):
    """
    List all trainable parameters with their shapes
    """
    print("\n=== Trainable Parameters ===")
    total_params = 0
    
    # Categorize parameters
    weight_matrices = []
    neuron_params = []
    other_params = []
    
    for name, param in net.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            
            print(f"{name}: {param.shape} ({param_count:,} parameters)")
            
            # Categorize based on parameter name patterns
            if any(keyword in name.lower() for keyword in ['weight', 'kernel']):
                weight_matrices.append((name, param.shape, param_count))
            elif any(keyword in name.lower() for keyword in ['decay', 'vth', 'threshold', 'amp']):
                neuron_params.append((name, param.shape, param_count))
            else:
                other_params.append((name, param.shape, param_count))
    
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Weight matrices: {len(weight_matrices)}")
    print(f"Neuron parameters: {len(neuron_params)}")
    print(f"Other parameters: {len(other_params)}")
    print("=" * 40)
    
    return weight_matrices, neuron_params, other_params

def load_and_inspect_parameters(h5_filepath):
    """
    Utility function to inspect the saved parameters
    
    :param h5_filepath: path to the H5 file
    """
    print(f"\n=== INSPECTING SAVED PARAMETERS: {h5_filepath} ===")
    
    with h5py.File(h5_filepath, 'r') as f:
        print("Available groups:")
        for group_name in f.keys():
            print(f"  - {group_name}")
            
            if group_name == 'training_metrics':
                group = f[group_name]
                print(f"    Training metrics datasets: {list(group.keys())}")
                print(f"    Configuration: {dict(group.attrs)}")
            else:
                group = f[group_name]
                print(f"    Parameters ({len(group.keys())} total):")
                print(f"    Timestamp: {group.attrs.get('timestamp', 'N/A')}")
                print(f"    Stage: {group.attrs.get('stage', 'N/A')}")
                
                # Show parameter summary
                weight_count = 0
                neuron_count = 0
                other_count = 0
                
                for param_name in group.keys():
                    param_data = group[param_name]
                    original_name = param_data.attrs.get('original_name', param_name)
                    
                    if any(keyword in original_name.lower() for keyword in ['weight', 'kernel']):
                        weight_count += 1
                    elif any(keyword in original_name.lower() for keyword in ['decay', 'vth', 'threshold', 'amp']):
                        neuron_count += 1
                    else:
                        other_count += 1
                
                print(f"    - Weight matrices: {weight_count}")
                print(f"    - Neuron parameters: {neuron_count}")
                print(f"    - Other parameters: {other_count}")

def compare_parameters_across_epochs(h5_filepath, param_name):
    """
    Compare how a specific parameter changes across epochs
    
    :param h5_filepath: path to the H5 file
    :param param_name: name of parameter to track (with dots replaced by underscores)
    """
    print(f"\n=== TRACKING PARAMETER: {param_name} ===")
    
    with h5py.File(h5_filepath, 'r') as f:
        epochs = []
        values = []
        
        # Check initial value
        if 'initial_parameters' in f and param_name in f['initial_parameters']:
            initial_val = f['initial_parameters'][param_name][:]
            print(f"Initial value shape: {initial_val.shape}")
            print(f"Initial value stats: min={initial_val.min():.6f}, max={initial_val.max():.6f}, mean={initial_val.mean():.6f}")
        
        # Collect values across epochs
        for group_name in f.keys():
            if group_name.startswith('epoch_'):
                epoch_num = int(group_name.split('_')[1])
                if param_name in f[group_name]:
                    param_val = f[group_name][param_name][:]
                    epochs.append(epoch_num)
                    values.append(param_val)
        
        # Sort by epoch
        sorted_data = sorted(zip(epochs, values))
        
        print(f"\nParameter evolution across {len(sorted_data)} epochs:")
        for epoch_num, param_val in sorted_data:
            print(f"Epoch {epoch_num:2d}: min={param_val.min():.6f}, max={param_val.max():.6f}, mean={param_val.mean():.6f}")
        
        # Check final value
        if 'final_parameters' in f and param_name in f['final_parameters']:
            final_val = f['final_parameters'][param_name][:]
            print(f"\nFinal value stats: min={final_val.min():.6f}, max={final_val.max():.6f}, mean={final_val.mean():.6f}")


# Unfreeze certain layers
def freeze_layers(model, args):
    """Freeze all layers except the one(s) specified in args."""

    # Map arg names to model layers
    layer_map = {
        "conv1": model.snn.conv1,
        "conv2": model.snn.conv2,
        "conv3": model.snn.conv3,
        "tc1":   model.snn.temp_conv1,
        "r1":    model.snn.rec1,
        "fc1":   model.snn.fc1,
        "fc2":   model.snn.fc2,
    }

    # Freeze all layers first
    for layer_name, layer in layer_map.items():
        for _, param in layer.named_parameters():
            param.requires_grad = False

    # Unfreeze only (weights) in the layer(s) given in args.unfreeze_layers
    # Example: args.unfreeze_layers = ["fc1", "fc2"]
    if hasattr(args, "unfreeze_layers") and args.unfreeze_layers:
        for layer_name in args.unfreeze_layers:
            if layer_name in layer_map:
                for name, param in layer_map[layer_name].named_parameters():
                        if "weight" in name:
                            param.requires_grad = True
                print(f">> {layer_name} unfrozen")
            else:
                print(f"!! Warning: {layer_name} not found in model.snn")


def fmt_float_for_name(x: float) -> str:
    # 0.05 -> "0p05"
    return str(x).replace(".", "p")


# Modified training function
def finetune_network_transfer_learning(dataset=EEGDataset2DLeftRight, network=WrapCUBASpikingCNN,
                                  dataset_kwargs=dict(), spike_ts=160, param_list=[],
                                  lr=[0.0001, 0.00001, 2*0.0001], weight_decays=[], 
                                  finetuning_batch_size=64, finetune_epochs=20,
                                  save_dir="training_results", seed=None, args=None):
    """
    Train SNN with transfer learning approach
    
    :param target_subject_id: Subject ID to use for transfer learning
    :param validation_ratio: Ratio of target subject data for validation (0.5 = 50%)
    :param pretrain_epochs: Number of epochs for pre-training
    :param finetune_epochs: Number of epochs for fine-tuning
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Start timing
    training_start_time = time.time()
    print(f"Transfer Learning Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Setup Dataset and split for transfer learning
    ds = dataset(**dataset_kwargs)
    folds = train_validate_split_5_fold_CV(ds)

    for fold, (all_finetune_indices, test_indices) in enumerate(folds, 1):
        random.seed(seed)
        finetune_indices, val_indices = [], []
        for cls in range(2):
            cls_indices = [i for i in all_finetune_indices if ds.label[i] == cls]
            random.shuffle(cls_indices)
        
            n_val = int(len(cls_indices) * 0.2)
            val_indices.extend(cls_indices[:n_val])
            finetune_indices.extend(cls_indices[n_val:])
        # Shuffle the final lists (optional but good for randomness)
        random.shuffle(finetune_indices)
        random.shuffle(val_indices)

        print("Finetuning Samples per Class: ")
        samples_per_class(ds.label[finetune_indices])
        print("Validation Samples per Class: ")
        samples_per_class(ds.label[val_indices])
        print("Test Samples per Class: ")
        samples_per_class(ds.label[test_indices])

        finetune_sampler = sampler.SubsetRandomSampler(finetune_indices)
        finetune_val_sampler = sampler.SubsetRandomSampler(val_indices)
        finetune_test_sampler = sampler.SubsetRandomSampler(test_indices)

        # Create data loaders
        finetune_loader = DataLoader(ds, batch_size=finetuning_batch_size, shuffle=False,
                                sampler=finetune_sampler, num_workers=4, pin_memory=True)
        finetune_val_loader = DataLoader(ds, batch_size=finetuning_batch_size, shuffle=False,
                            sampler=finetune_val_sampler, num_workers=4, pin_memory=True)
        finetune_test_loader = DataLoader(ds, batch_size=finetuning_batch_size, shuffle=False,
                            sampler=finetune_test_sampler, num_workers=4, pin_memory=True)
        
        # Setup Network
        pattern = (f"snn_experiment_results/pretrained_model_5_fold/snn_pretrained_model_with_non_selected_participants_seed_5_fold_{fold}_*.pth")
        matches = glob.glob(pattern)
        pretrained_model_path = sorted(matches)[-1]
        print("Loading:", pretrained_model_path)
        net = WrapCUBASpikingCNN(spike_ts=spike_ts, device=device, param_list=param_list)
        net.load_state_dict(torch.load(pretrained_model_path, map_location=device))

        x = getattr(args, 'x', 4.0)
        quantize_all_weights(net, q=2) # Hard coded for 2 levels, change for generalized case (todo)
        add_gaussian_noise_levelwise(net, x=x, seed=5)
        net.to(device)
        net.train()

        base_range_raw = {}
        with torch.no_grad():
            for name, p in net.named_parameters():
                if name.endswith(".weight"):
                    if name == "snn.conv1.psp_func.weight":
                        max = 0.3330
                    elif name == "snn.conv2.psp_func.weight":
                        max = 0.0417
                    elif name == "snn.conv3.psp_func.weight":
                        max = 0.0295
                    elif name == "snn.temp_conv1.psp_func_list.0.weight":
                        max = 0.0625
                    elif name == "snn.temp_conv1.psp_func_list.1.weight":
                        max = 0.0625
                    elif name == "snn.temp_conv1.psp_func_list.2.weight":
                        max = 0.0625
                    elif name == "snn.rec1.rec_func.weight":
                        max = 0.0625
                    elif name == "snn.fc1.psp_func.weight":
                        max = 0.0625
                    elif name == "snn.fc2.weight":
                        max = 0.0625
                    else:
                        max = None
                        print(f"Warning: No A value found for layer {name}, using max abs value.")
                    base_range_raw[name] = float(2.0 * max)

        # Unfreeze selected layer. 
        if args is not None:
            freeze_layers(net, args)

        # List and save initial parameters
        # Create H5 file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # h5_filepath = os.path.join(save_dir, f"snn_transfer_learning_finetuning_seed_{seed}_fold_{fold}_{timestamp}.h5")

        print("=== INITIAL NETWORK PARAMETERS ===")
        # save_parameters_to_h5(net, h5_filepath, stage="begin", save_interval=1)
        # print(f"Initial parameters saved to {h5_filepath}")

        # Setup loss function and optimizer parameters
        criterion = nn.CrossEntropyLoss()

        decays = ['snn.c1_vdecay', 'snn.c2_vdecay', 'snn.c3_vdecay', 'snn.tc1_vdecay',
                'snn.tc1_cdecay', 'snn.r1_vdecay', 'snn.f1_vdecay', 'snn.c1_cdecay',
                'snn.c2_cdecay', 'snn.c3_cdecay', 'snn.r1_cdecay', 'snn.f1_cdecay']

        ts_weights = ['snn.ts_weights']

        # freeze all ts
        for name, p in net.named_parameters():
            if name in ts_weights:
                p.requires_grad = False

        weight_matrices, neuron_params, other_params = list_parameters(net)

        # decay_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in decays, net.named_parameters()))))
        # ts_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in ts_weights, net.named_parameters()))))
        decay_params = [p for name, p in net.named_parameters() if name in decays and p.requires_grad]
        ts_params    = [p for name, p in net.named_parameters() if name in ts_weights and p.requires_grad]
        # weights = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in decays+ts_weights, net.named_parameters()))))
        trainable = [(name, p) for name, p in net.named_parameters() if (name not in decays + ts_weights) and p.requires_grad]

        weights = [p for name, p in trainable]
        weight_names = {name for name, p in trainable}

        # exclude frozen layers
        base_range = {}
        with torch.no_grad():
            for name, p in net.named_parameters():
                if name in weight_names and name in base_range_raw:
                    base_range[p] = base_range_raw[name]
        
        optimizer = optim.Adam([{'params': weights}, {'params': decay_params, 'lr': lr[0]}, {'params': ts_params, 'lr': lr[1]}], lr=lr[2])
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 20,
            eta_min=1e-5,
        )

        dut = getattr(args, 'dut', "dut2")
        rsd_c2c = getattr(args, "rsd_c2c", 0.0)
        sigma_add = getattr(args, "sigma_add", 0.0)
        sigma_mul = getattr(args, "sigma_mul", 0.0)
        q = getattr(args, "q", 0)

        thr_str  = fmt_float_for_name(getattr(args, "thr", 0.05))
        thr_asym_str = fmt_float_for_name(getattr(args, "thr_asym", 1.0))
        A_scale_str = fmt_float_for_name(getattr(args, "A_scale", 1.0))
        A_asym_str = fmt_float_for_name(getattr(args, "A_asym", 1.0))
        rsd_str = fmt_float_for_name(rsd_c2c)
        sigma_add_str = fmt_float_for_name(getattr(args, "sigma_add", 0.0))
        sigma_mul_str = fmt_float_for_name(getattr(args, "sigma_mul", 0.0))
        
        # Create accuracy log file
        accuracy_file = os.path.join(save_dir, f"accuracy_per_epoch_seed_{seed}_fold_{fold}_dut_{dut}_thr_{thr_str}_thrasym_{thr_asym_str}_Ascale_{A_scale_str}_Aasym_{A_asym_str}_rsdc2c_{rsd_str}_q_{q}_{timestamp}.txt")
        with open(accuracy_file, 'w') as f:
            f.write("Phase,Epoch,Train_Loss,Val_Accuracy,Class0_Accuracy,Class1_Accuracy,Test_Accuracy,learning_rate\n")

        # Accuracy log file (per sample)
        acc_over_samples_file = os.path.join(save_dir, f"accuracy_per_sample_seed_{seed}_fold_{fold}_dut_{dut}_thr_{thr_str}_thrasym_{thr_asym_str}_Ascale_{A_scale_str}_Aasym_{A_asym_str}_rsdc2c_{rsd_str}_q_{q}_{timestamp}.txt")
        with open(acc_over_samples_file, 'w') as f:
            f.write("Phase,Epoch,BatchIdx,SamplesSeen,Val_Accuracy,Class0_Accuracy,Class1_Accuracy,Test_Accuracy,LTP_crossings,LTD_crossings\n")

        # Initial acc test after quantization
        net.eval()
        acc_step, class_acc_step = test_accuracy(net, finetune_val_loader, device)
        acc_test_step, _ = test_accuracy(net, finetune_test_loader, device)
        net.train()

        with open(acc_over_samples_file, 'a') as f:
            f.write(f"finetune,{0},{0},{0},{acc_step:.6f},{class_acc_step[0]:.6f},{class_acc_step[1]:.6f},{acc_test_step:.6f}\n")

        # Parameters for quantization
        thr = getattr(args, "thr", 0.05)   # + step
        w_acc = {p: torch.zeros_like(p.data) for p in weights}   # accumulate weight updates

        def beta_kernel_c2c(w, A, alpha, beta, rsd=0.0, sigma_add=0.0, sigma_mul=0.0):
            """
            Beta kernel with cycle-to-cycle variation on A.
            A_eff ~ N(A, (rsd*A)^2) each time this function is called.
            sigma_add: additive noise stddev
            sigma_mul: multiplicative noise stddev
            0.0 means no noise
            """
            # print(f"Shape of w in beta_kernel_c2c: {w.shape}")
            if rsd > 0.0:
                # Sample A_eff for each element in w
                A_eff = A + torch.randn_like(w) * (rsd * A)
            else:
                A_eff = A
            dW = A_eff * (w**(alpha - 1)) * ((1 - w)**(beta - 1))
            if sigma_add > 0.0:
                dW += torch.randn_like(w) * sigma_add
                #Clamp between -1 and 1
                dW = torch.clamp(dW, -1.0, 1.0)
            if sigma_mul > 0.0:
                dW *= (1.0 + torch.randn_like(w) * sigma_mul)
                #Clamp between -1 and 1
                dW = torch.clamp(dW, -1.0, 1.0)
            return dW

        
        # model parameters 25 mV
        if dut == "dut1":
            A_ltp = 0.1761
            A_ltd = 0.3300
            A_scale = getattr(args, 'A_scale', 1.0)
            A_asym = getattr(args, 'A_asym', 1.0)
            alpha_ltp = 1.8103
            alpha_ltd = 2.4721
            beta_ltp = 2.1228
            beta_ltd = 1.7929 

        # model parameters 50 mV
        elif dut == "dut2":
            A_ltp = 0.2611
            A_ltd = 0.5969
            A_scale = getattr(args, 'A_scale', 1.0)
            A_asym = getattr(args, 'A_asym', 1.0)
            alpha_ltp = 1.6429
            alpha_ltd = 2.4585
            beta_ltp = 1.9873
            beta_ltd = 1.6885
        # DUT3 LTD parameters (A, alpha, beta): [0.10299356 2.05557189 1.77256752] LTP parameters: [0.24897956 1.86338502 2.91746585] Pulse width: 2e-08
        # DUT4 LTD parameters: [0.18900954 2.40094446 1.97854856] LTP parameters: [0.36361457 1.96391651 3.09212352] Pulse width: 2e-07
        # DUT5 LTD parameters: [0.36568703 2.75019033 2.2682048 ] LTP parameters: [0.65606012 1.90866458 3.47689647] Pulse width: 2e-05
        elif dut == "dut3":
            A_ltp = 0.24897956
            A_ltd = 0.10299356
            A_scale = getattr(args, 'A_scale', 1.0)
            A_asym = getattr(args, 'A_asym', 1.0)
            alpha_ltp = 1.86338502
            alpha_ltd = 2.05557189
            beta_ltp = 2.91746585
            beta_ltd = 1.77256752
        elif dut == "dut4":
            A_ltp = 0.36361457
            A_ltd = 0.18900954
            A_scale = getattr(args, 'A_scale', 1.0)
            A_asym = getattr(args, 'A_asym', 1.0)
            alpha_ltp = 1.96391651
            alpha_ltd = 2.40094446
            beta_ltp = 3.09212352
            beta_ltd = 1.97854856
        elif dut == "dut5":
            A_ltp = 0.65606012
            A_ltd = 0.36568703
            A_scale = getattr(args, 'A_scale', 1.0)
            A_asym = getattr(args, 'A_asym', 1.0)
            alpha_ltp = 1.90866458
            alpha_ltd = 2.75019033
            beta_ltp = 3.47689647
            beta_ltd = 2.2682048

        # ==================== PHASE 1: PRE-TRAINING ====================
        print(f"\n=== PHASE 1: FINETUNING FOR {finetune_epochs} EPOCHS ===")
        print("Training on all subjects except target subject...")
        
        finetune_start_time = time.time()

        best_val_acc = 0.0
        best_finetune_state = None
        
        samples_seen = 0 

        for e in range(1, finetune_epochs+1):
            running_loss = 0
            train_ita = 0

            updates_total = 0
            ltp_crossings = 0 
            ltd_crossings = 0

            num_batches = len(finetune_loader)
            log_points = np.linspace(1, num_batches, 10, dtype=int)  # exactly 10 indices        
            
            for i, data in enumerate(finetune_loader, 0):
                eeg_data, label = data
                eeg_data, label = eeg_data.to(device), label.to(device)
                optimizer.zero_grad()
                output = net(eeg_data)
                loss = criterion(output, label)
                loss.backward()
                w_old = {p: p.data.clone() for p in weights}
                optimizer.step()

                with torch.no_grad():
                    for p in weights:
                        # continuous update from Adam for this batch
                        raw = p.data - w_old[p]

                        # accumulate into w_acc
                        acc = w_acc[p] + raw

                        init_range = base_range[p]
                        thr_update = thr * init_range
                        thr_ltp = thr_update
                        thr_ltd = thr_update * float(args.thr_asym)
 
                        # number of positive / negative threshold crossings
                        # threshold is |w_acc| >= thr * init_range
                        pos = torch.floor(acc / thr_ltp)
                        pos = torch.clamp(pos, min=0)  # only positive counts
                        if (pos > 1.0).any().item(): print("Large pos backprop update")

                        neg = torch.floor(-acc / thr_ltd) 
                        neg = torch.clamp(neg, min=0)  # only positive counts, for negative side
                        if (neg > 1.0).any().item(): print("Large neg backprop update")

                        def w_normalized(w):
                            return (w + init_range / 2) / init_range
                        
                        wn = torch.clamp(w_normalized(w_old[p]), 0.01, 0.99)
                        dw_pos = beta_kernel_c2c(wn, A_ltp * A_scale, alpha_ltp, beta_ltp, rsd=rsd_c2c, sigma_add=sigma_add, sigma_mul=sigma_mul) * init_range
                        dw_neg = beta_kernel_c2c(wn, A_ltd * A_scale * A_asym, alpha_ltd, beta_ltd, rsd=rsd_c2c, sigma_add=sigma_add, sigma_mul=sigma_mul) * init_range
                        p.data = w_old[p] + dw_pos * (pos > 0).to(w_old[p].dtype) - dw_neg * (neg > 0).to(w_old[p].dtype)
                        p.data.clamp_(-init_range / 2, init_range / 2)

                        # update accumulator: each crossing uses up thr_update/thr_update*thr_asym in the accumulator
                        acc = acc - pos * thr_ltp + neg * thr_ltd
                        w_acc[p] = acc

                        # stats
                        n_pos = int(pos.sum().item())
                        n_neg = int(neg.sum().item())
                        updates_total += (n_pos + n_neg)

                        # count LTP / LTD separately
                        ltp_crossings += n_pos
                        ltd_crossings += n_neg

                        # print(p.data == w_old[p])

                    if (q>0):
                        quantize_all_weights(net, q=q)

                running_loss += loss.to('cpu').item()
                train_ita = i

                batch_size_now = label.size(0)
                samples_seen += batch_size_now

                if (i+1) in log_points:
                    net.eval()
                    acc_step, class_acc_step = test_accuracy(net, finetune_val_loader, device)
                    acc_test_step, _ = test_accuracy(net, finetune_test_loader, device)
                    net.train()

                    with open(acc_over_samples_file, 'a') as f:
                        f.write(f"finetune,{e},{i+1},{samples_seen},{acc_step:.6f},{class_acc_step[0]:.6f},{class_acc_step[1]:.6f},{acc_test_step:.6f},{ltp_crossings},{ltd_crossings}\n")

            # print statistics
            print(f"[epoch {e}] LTP crossings: {ltp_crossings}, LTD crossings: {ltd_crossings}")

            # Clamp decay parameters
            for param in decay_params:
                param.data = param.data.clamp(min=1e-7)

            # Evaluate on 20% of data
            net.eval()
            acc_val, class_acc_val = test_accuracy(net, finetune_val_loader, device)
            acc_test, _ = test_accuracy(net, finetune_test_loader, device)
            net.train()

            avg_loss = running_loss / (train_ita + 1)
            print('Pre-train Epoch: %d, Loss: %.3f' % (e, avg_loss))
            print('Validation Accuracy after Pre-train Epoch %d: %.3f %%' % (e, acc_val * 100))
            print('Validation Accuracy for class 0: %.3f %%, 1: %.3f %%' %
                (class_acc_val[0]*100, class_acc_val[1]*100))
            print('Test Accuracy after Pre-train Epoch %d: %.3f %%' % (e, acc_test * 100))
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"lr: {current_lr}")

            # Save accuracy to file
            with open(accuracy_file, 'a') as f:
                f.write(f"finetune,{e},{avg_loss:.6f},{acc_val:.6f},{class_acc_val[0]:.6f},{class_acc_val[1]:.6f},{acc_test:.6f},{current_lr:.8f}\n")

            scheduler.step()

            # ---- Save the best model ----
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_finetune_state = copy.deepcopy(net.state_dict())


            # Save parameters after every 3 finetuning epoch OR if it's the final epoch
            # save_parameters_to_h5(net, h5_filepath, epoch=e, stage="finetune", save_interval=1)

            
        # Restore best pretrained model
        if best_finetune_state:
            net.load_state_dict(best_finetune_state)
        
        # Save pre-trained model
        if best_finetune_state:
            finetune_weights_path = os.path.join(save_dir, f"snn_finetuned_model_seed_{seed}_fold_{fold}_dut_{dut}_thr_{thr_str}_thrasym_{thr_asym_str}_Ascale_{A_scale_str}_Aasym_{A_asym_str}_rsdc2c_{rsd_str}_q_{q}_{timestamp}.pth")
            torch.save(best_finetune_state, finetune_weights_path)
            print(f"Finetuned model saved to {finetune_weights_path}")

        finetune_time = time.time() - finetune_start_time
        print(f"Fine-tune Time: {finetune_time:.2f} seconds ({finetune_time/60:.2f} minutes)")

            
        # Save timing information to H5 file
        # with h5py.File(h5_filepath, 'a') as f:
        #     timing_group = f.create_group('transfer_learning_metrics')
        #     timing_group.create_dataset('finetune_time', data=finetune_time)
        #     timing_group.attrs['device'] = str(device)
        #     timing_group.attrs['finetuning_batch_size'] = finetuning_batch_size
        #     timing_group.attrs['finetune_epochs'] = finetune_epochs

    
    return net #, h5_filepath


if __name__ == "__main__":

    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    """
    Dataset Parameters
    """
    DATASET = EEGDataset2DLeftRight 
    USE_IMAGERY = True

    # Define subset of subjects for LOOCV
    SUBJECT_LIST = [i + 1 for i in range(109)]
    EXCLUDED = [88, 89, 92, 100, 104, 106]    # subjects to exclude
    SUBJECT_LIST = [s for s in SUBJECT_LIST if s not in EXCLUDED]

    ds_params = {
        "base_route": str(_REPO_ROOT / "dataset" / "eegmmidb_slice_norm") + "/",
        "subject_id_list": SUBJECT_LIST,
        "start_ts": 0,
        "end_ts": 161,
        "window_ts": 160,
        "overlap_ts": 0,
        "use_imagery": USE_IMAGERY,
        "transform": ToTensor()
    }

    SPIKE_TS = 160
    BATCH_SIZE = 64
    WT_LR = 0.0001
    TS_LR = 0.0001
    NEURON_LR = 0.0001
    N_CLASSES = 2
    VDECAY = 0.1
    CDECAY = 0.1
    VTH = 0.1
    GRAD_WIN = 0.3
    TH_AMP = 0.01
    TH_DECAY = 0.1
    BASE_TH = 0.01
    WT_DECAY = 1e-6
    TS_DECAY = 2 * WT_DECAY
    NEURON_DECAY = 2 * WT_DECAY
    PARAM_LIST = [CDECAY, VDECAY, VTH, GRAD_WIN, TH_AMP, TH_DECAY, BASE_TH]
    
    
    print("Starting SNN finetuning with parameter freezing for transfer learning")        

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU only")
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

    # save_dir = f"snn_experiment_results/finetuned_model_5_fold_unfrozen_{args.unfreeze_layers}_model"
    x = getattr(args, 'x', 4.0)
    save_dir = f"snn_experiment_results/retuned_model_5_fold_all_layers_model/x_{x}/epochs_{args.retuning_epochs}"

    net = finetune_network_transfer_learning(             # net, h5_file = ...
        dataset=DATASET,
        dataset_kwargs=ds_params,
        spike_ts=SPIKE_TS,
        param_list=PARAM_LIST,
        finetuning_batch_size=args.retuning_batch_size,
        finetune_epochs=args.retuning_epochs,
        lr=[args.lr, args.lr, args.lr],
        weight_decays=[args.neuron_decay, args.ts_decay, args.wt_decay],
        save_dir=save_dir,
        seed = args.seed,
        args=args
    )

    print(f"\n=== TRAINING COMPLETED ===")
   
