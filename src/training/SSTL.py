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
from utility import samples_per_class, train_validate_split_4_fold_CV_transfer_learning
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
                for _, param in layer_map[layer_name].named_parameters():
                    param.requires_grad = True
                print(f">> {layer_name} unfrozen")
            else:
                print(f"!! Warning: {layer_name} not found in model.snn")

def fmt_float_for_name(x: float) -> str:
    # 0.05 -> "0p05"
    return str(x).replace(".", "p")

# Modified training function
def finetune_network_transfer_learning(dataset=EEGDataset2DLeftRight, pretrained_model_path=None, target_subject_id=1,                                
                                  dataset_kwargs=dict(),
                                  finetune_lr=[], 
                                  finetuning_batch_size=1, finetune_epochs=1,
                                  save_dir="training_results", seed=None, args=None):
    """
    Train SNN with transfer learning approach
    
    :param target_subject_id: Subject ID to use for transfer learning
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
    folds = train_validate_split_4_fold_CV_transfer_learning(ds, target_subject_id, seed=seed)

    for fold, (finetune_indices, val_indices) in enumerate(folds, 1):
        

        print("Finetuning Samples per Class: ")
        samples_per_class(ds.label[finetune_indices])

        print("Test Samples per Class: ")
        samples_per_class(ds.label[val_indices])

        finetune_sampler = sampler.SubsetRandomSampler(finetune_indices)
        finetune_val_sampler = sampler.SubsetRandomSampler(val_indices)

        # Create data loaders
        finetune_loader = DataLoader(ds, batch_size=finetuning_batch_size, shuffle=False,
                                sampler=finetune_sampler, num_workers=4, pin_memory=True)
        finetune_val_loader = DataLoader(ds, batch_size=finetuning_batch_size, shuffle=False,
                            sampler=finetune_val_sampler, num_workers=4, pin_memory=True)
        
        # Setup Network
        SPIKE_TS = 160
        PARAM_LIST = [0.1, 0.1, 0.1, 0.3, 0.01, 0.1, 0.01]

        net = WrapCUBASpikingCNN(spike_ts=SPIKE_TS, device=device, param_list=PARAM_LIST)
        net.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        # move model to device
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
        h5_filepath = os.path.join(save_dir, f"snn_transfer_learning_finetuning_seed_{seed}_fold_{fold}_{timestamp}.h5")

        print("=== INITIAL NETWORK PARAMETERS ===")
        save_parameters_to_h5(net, h5_filepath, stage="begin", save_interval=1)
        print(f"Initial parameters saved to {h5_filepath}")

        # Setup loss function and optimizer parameters
        criterion = nn.CrossEntropyLoss()

        decays = ['snn.c1_vdecay', 'snn.c2_vdecay', 'snn.c3_vdecay', 'snn.tc1_vdecay',
                'snn.tc1_cdecay', 'snn.r1_vdecay', 'snn.f1_vdecay', 'snn.c1_cdecay',
                'snn.c2_cdecay', 'snn.c3_cdecay', 'snn.r1_cdecay', 'snn.f1_cdecay']

        ts_weights = ['snn.ts_weights']

        # # freeze all ts
        # for name, p in net.named_parameters():
        #     if name in ts_weights:
        #         p.requires_grad = False

        weight_matrices, neuron_params, other_params = list_parameters(net)

        # decay_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in decays, net.named_parameters()))))
        # ts_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in ts_weights, net.named_parameters()))))
        decay_params = [p for name, p in net.named_parameters() if name in decays and p.requires_grad]
        ts_params    = [p for name, p in net.named_parameters() if name in ts_weights and p.requires_grad]
        # weights = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in decays+ts_weights, net.named_parameters()))))
        memristive_params = [(name, p) for name, p in net.named_parameters() if ('weight' in name) and (name not in decays + ts_weights) and p.requires_grad]
        weights = [p for name, p in memristive_params]
        weight_names = {name for name, p in memristive_params}

        # exclude frozen layers
        base_range = {}
        with torch.no_grad():
            for name, p in memristive_params:
                if name in weight_names and name in base_range_raw:
                    base_range[p] = base_range_raw[name]
        
        optimizer = optim.Adam([{'params': weights}, {'params': decay_params, 'lr': finetune_lr[0]}, {'params': ts_params, 'lr': finetune_lr[1]}], lr=finetune_lr[2])
        



        dWmin_str  = fmt_float_for_name(getattr(args, "thr", 0.05))
        A_scale_str = fmt_float_for_name(getattr(args, "A_scale", 1.0))
        A_asym_str = fmt_float_for_name(getattr(args, "A_asym", 1.0))


        dW_min = getattr(args, 'thr', 0.05)   # + step
        w_acc = {p: torch.zeros_like(p.data) for p in weights}   # accumulate weight updates
        
        # Create accuracy log file
        accuracy_file = os.path.join(save_dir, f"transfer_learning_finetuning_accuracy_seed_{seed}_fold_{fold}_dW_min_{dW_min}_{timestamp}.txt")
        with open(accuracy_file, 'w') as f:
            f.write("Phase,Epoch,Train_Loss,Val_Accuracy,Class0_Accuracy,Class1_Accuracy,learning_rate,LTP_crossings,LTD_crossings\n")


        def beta_kernel(w, A, alpha, beta):
            return A * (w**(alpha - 1)) * ((1 - w)**(beta - 1))
        
        dut = getattr(args, 'dut', "dut2")
        
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

        # ==================== PHASE 1: PRE-TRAINING ====================
        print(f"\n=== PHASE 1: FINETUNING FOR {finetune_epochs} EPOCHS ===")
        
        finetune_start_time = time.time()

        
        samples_seen = 0 

        for e in range(1, finetune_epochs+1):
            running_loss = 0
            train_ita = 0

            updates_total = 0
            ltp_crossings = 0 
            ltd_crossings = 0

            num_batches = len(finetune_loader)
         
            
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

                        w_range = base_range[p]

                        thr = dW_min * w_range
 
                        # number of positive / negative threshold crossings
                        # threshold is |w_acc| >= dW_min * w_range
                        pos = torch.floor(acc / thr)
                        pos = torch.clamp(pos, min=0)  # only positive counts
                        if (pos > 1.0).any().item(): print("Large pos backprop update")

                        neg = torch.floor(-acc / thr) 
                        neg = torch.clamp(neg, min=0)  # only positive counts, for negative side
                        if (neg > 1.0).any().item(): print("Large neg backprop update")

                        def w_normalized(w):
                            return (w + w_range / 2) / w_range
                        
                        wn = torch.clamp(w_normalized(w_old[p]), 0.01, 0.99)
                        dW_pos = beta_kernel(wn, A_ltp*A_scale, alpha_ltp, beta_ltp) * w_range
                        dW_neg = beta_kernel(wn, A_ltd*A_scale*A_asym, alpha_ltd, beta_ltd) * w_range
                        p.data = w_old[p] + dW_pos * (pos > 0).to(w_old[p].dtype) - dW_neg * (neg > 0).to(w_old[p].dtype)

                        p.data.clamp_(-w_range / 2, w_range / 2)

                        # update accumulator: each crossing uses up ±dW_min in the accumulator
                        acc = acc - pos * thr + neg * thr
                        w_acc[p] = acc

                        # stats
                        n_pos = int(pos.sum().item())
                        n_neg = int(neg.sum().item())
                        updates_total += (n_pos + n_neg)

                        # count LTP / LTD separately
                        ltp_crossings += n_pos
                        ltd_crossings += n_neg

                        # print(p.data == w_old[p])
                

                running_loss += loss.to('cpu').item()
                train_ita = i

                batch_size_now = label.size(0)
                samples_seen += batch_size_now

            # print statistics
            print(f"[epoch {e}] LTP crossings: {ltp_crossings}, LTD crossings: {ltd_crossings}")

            # Clamp decay parameters
            for param in decay_params:
                param.data = param.data.clamp(min=1e-7)

            # Evaluation
            net.eval()
            acc_val, class_acc_val = test_accuracy(net, finetune_val_loader, device)
            net.train()

            avg_loss = running_loss / (train_ita + 1)
            print('Pre-train Epoch: %d, Loss: %.3f' % (e, avg_loss))
            print('Validation Accuracy after Pre-train Epoch %d: %.3f %%' % (e, acc_val * 100))
            print('Validation Accuracy for class 0: %.3f %%, 1: %.3f %%' %
                (class_acc_val[0]*100, class_acc_val[1]*100))
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"lr: {current_lr}")

            # Save accuracy to file
            with open(accuracy_file, 'a') as f:
                f.write(f"finetune,{e},{avg_loss:.6f},{acc_val:.6f},{class_acc_val[0]:.6f},{class_acc_val[1]:.6f},{current_lr:.8f},{ltp_crossings},{ltd_crossings}\n")

        finetune_weights_path = os.path.join(save_dir, f"snn_finetuned_model_seed_{seed}_fold_{fold}_dWmin_{dWmin_str}_Ascale_{A_scale_str}_Aasym_{A_asym_str}_{timestamp}.pth")
        torch.save(net, finetune_weights_path)
        print(f"Finetuned model saved to {finetune_weights_path}")

        finetune_time = time.time() - finetune_start_time
        print(f"Fine-tune Time: {finetune_time:.2f} seconds ({finetune_time/60:.2f} minutes)")

            
        # Save timing information to H5 file
        with h5py.File(h5_filepath, 'a') as f:
            timing_group = f.create_group('transfer_learning_metrics')
            timing_group.create_dataset('finetune_time', data=finetune_time)
            timing_group.attrs['device'] = str(device)
            timing_group.attrs['finetuning_batch_size'] = finetuning_batch_size
            timing_group.attrs['finetune_epochs'] = finetune_epochs

    
    return net, h5_filepath


if __name__ == "__main__":

    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    """
    Dataset Parameters
    """
    DATASET = EEGDataset2DLeftRight 
    USE_IMAGERY = True

    target_subject_lists = [[i + 1 for i in range(21)],
                            [i + 1 for i in range(21, 42)],
                            [i + 1 for i in range (42, 63)],
                            [i + 1 for i in range (63, 83)],
                            [84, 85, 86, 87, 90, 91, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 105, 107, 108, 109]
                           ]
    
    # Put the name of pretrained model
    pretrained_models = ['model_online_seed_5_fold_1_dut_dut2_thr_0p025_thrasym_1p0_Ascale_1p0_Aasym_1p0_rsdc2c_0.0_20251127_215126.pth',
                         'model_online_seed_5_fold_2_dut_dut2_thr_0p025_thrasym_1p0_Ascale_1p0_Aasym_1p0_rsdc2c_0.0_20251127_222122.pth',
                         'model_online_seed_5_fold_3_dut_dut2_thr_0p025_thrasym_1p0_Ascale_1p0_Aasym_1p0_rsdc2c_0.0_20251127_225119.pth',
                         'model_online_seed_5_fold_4_dut_dut2_thr_0p025_thrasym_1p0_Ascale_1p0_Aasym_1p0_rsdc2c_0.0_20251127_232115.pth',
                         'model_online_seed_5_fold_5_dut_dut2_thr_0p025_thrasym_1p0_Ascale_1p0_Aasym_1p0_rsdc2c_0.0_20251127_235115.pth'
                         ]

    

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

    for k in range(5):

        target_subject_list = target_subject_lists[k]

        print(f'Target subject list: {target_subject_list}')

        pretrained_model = pretrained_models[k]

        print(f'Pretrained model: {pretrained_model}')

        ds_params = {
            "base_route": str(_REPO_ROOT / "dataset" / "eegmmidb_slice_norm") + "/",
            "subject_id_list": target_subject_list,
            "start_ts": 0,
            "end_ts": 161,
            "window_ts": 160,
            "overlap_ts": 0,
            "use_imagery": USE_IMAGERY,
            "transform": ToTensor()
        }

        for i in target_subject_list:
            target_subject_id = i
            print("Starting SNN finetuning with parameter tracking for 3-fold CV")        

            unfrozen_str = "".join(args.unfreeze_layers) if args.unfreeze_layers else "none"
            dW_min_str = args.thr

            save_dir = f"snn_experiment_results/subject_specific_4_fold_transfer_learning_full_online_device_batchsize_{args.finetuning_batch_size}_with_ts/fold_{k+1}/layer_unfrozen_{unfrozen_str}_lr_{args.finetune_lr}_dW_min_{dW_min_str}/subject_{target_subject_id}"

            # Path to  savedpretrained model for SSTL
            pretrained_model_path = f"snn_experiment_results/SSTL_pretrained_model/{pretrained_model}"

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            print(f"\n=== Training with validation subject {target_subject_id} Seed {args.seed}===")

            net, h5_file = finetune_network_transfer_learning(
                dataset=DATASET,
                dataset_kwargs=ds_params,
                pretrained_model_path=pretrained_model_path,
                target_subject_id=target_subject_id,
                finetuning_batch_size=args.finetuning_batch_size,
                finetune_epochs=args.finetune_epochs,
                finetune_lr=[args.finetune_lr, args.finetune_lr, args.finetune_lr],
                save_dir=save_dir,
                seed = args.seed,
                args=args
            )

    print(f"\n=== TRAINING COMPLETED ===")
   
