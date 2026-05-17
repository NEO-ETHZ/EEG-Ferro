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

def freeze_parameters(model, args):

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

    for parameter in args.freeze_parameters:
        if parameter:
            for layer_name, layer in layer_map.items():
                for param_name, param in layer.named_parameters():
                    if parameter == param_name:
                        param.requires_grad = False
                        print(f"Frozen parameter '{param_name}' in layer '{layer_name}")


# Modified training function
def pretrain_network_transfer_learning(dataset=EEGDataset2DLeftRight, network=WrapCUBASpikingCNN,
                                  dataset_kwargs=dict(), spike_ts=160, param_list=[],
                                  lr=[0.0001, 0.00001, 2*0.0001], weight_decays=[], 
                                  pretraining_batch_size=64, pretrain_epochs=20,
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

    for fold, (all_pretrain_indices, test_indices) in enumerate(folds, 1):
        random.seed(seed)
        pretrain_indices, val_indices = [], []
        for cls in range(2):
            cls_indices = [i for i in all_pretrain_indices if ds.label[i] == cls]
            random.shuffle(cls_indices)
        
            n_val = int(len(cls_indices) * 0.2)
            val_indices.extend(cls_indices[:n_val])
            pretrain_indices.extend(cls_indices[n_val:])
        # Shuffle the final lists (optional but good for randomness)
        random.shuffle(pretrain_indices)
        random.shuffle(val_indices)



        print("Pre-training Samples per Class: ")
        samples_per_class(ds.label[pretrain_indices])
        print("Validation Samples per Class: ")
        samples_per_class(ds.label[val_indices])
        print("Test Samples per Class: ")
        samples_per_class(ds.label[test_indices])

        pretrain_sampler = sampler.SubsetRandomSampler(pretrain_indices)
        pretrain_val_sampler = sampler.SubsetRandomSampler(val_indices)
        pretrain_test_sampler = sampler.SubsetRandomSampler(test_indices)

        # Create data loaders
        pretrain_loader = DataLoader(ds, batch_size=pretraining_batch_size, shuffle=False,
                                sampler=pretrain_sampler, num_workers=4, pin_memory=True)
        pretrain_val_loader = DataLoader(ds, batch_size=pretraining_batch_size, shuffle=False,
                            sampler=pretrain_val_sampler, num_workers=4, pin_memory=True)
        pretrain_test_loader = DataLoader(ds, batch_size=pretraining_batch_size, shuffle=False,
                            sampler=pretrain_test_sampler, num_workers=4, pin_memory=True)
        
        # Setup Network
        net = network(spike_ts, device, param_list=param_list)
        net.to(device)
        net.train()

        # List and save initial parameters
        # Create H5 file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        h5_filepath = os.path.join(save_dir, f"snn_transfer_learning_pretraining_seed_{seed}_fold_{fold}_{timestamp}.h5")

        print("=== INITIAL NETWORK PARAMETERS ===")
        weight_matrices, neuron_params, other_params = list_parameters(net)
        save_parameters_to_h5(net, h5_filepath, stage="begin", save_interval=1)
        print(f"Initial parameters saved to {h5_filepath}")

        # Setup loss function and optimizer parameters
        criterion = nn.CrossEntropyLoss()

        decays = ['snn.c1_vdecay', 'snn.c2_vdecay', 'snn.c3_vdecay', 'snn.tc1_vdecay',
                'snn.tc1_cdecay', 'snn.r1_vdecay', 'snn.f1_vdecay', 'snn.c1_cdecay',
                'snn.c2_cdecay', 'snn.c3_cdecay', 'snn.r1_cdecay', 'snn.f1_cdecay']

        ts_weights = ['snn.ts_weights']

        decay_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in decays, net.named_parameters()))))
        ts_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in ts_weights, net.named_parameters()))))
        weights = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in decays+ts_weights, net.named_parameters()))))
        
        optimizer = optim.Adam([{'params': weights}, {'params': decay_params, 'lr': lr[0]}, {'params': ts_params, 'lr': lr[1]}], lr=lr[2])
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 20,
            eta_min=1e-5,
        )

        # Create accuracy log file
        accuracy_file = os.path.join(save_dir, f"transfer_learning_pretraining_accuracy_seed_{seed}_fold_{fold}_{timestamp}.txt")
        with open(accuracy_file, 'w') as f:
            f.write("Phase,Epoch,Train_Loss,Val_Accuracy,Class0_Accuracy,Class1_Accuracy,Test_Accuracy,learning_rate\n")

        # Accuracy log file (per sample)
        acc_over_samples_file = os.path.join(save_dir, f"transfer_learning_pretraining_accuracy_over_samples_seed_{seed}_fold_{fold}_{timestamp}.txt")
        with open(acc_over_samples_file, 'w') as f:
            f.write("Phase,Epoch,BatchIdx,SamplesSeen,Val_Accuracy,Class0_Accuracy,Class1_Accuracy,Test_Accuracy\n")

        # ==================== PHASE 1: PRE-TRAINING ====================
        print(f"\n=== PHASE 1: PRE-TRAINING FOR {pretrain_epochs} EPOCHS ===")
        print("Training on all subjects except target subject...")
        
        pretrain_start_time = time.time()

        best_val_acc = 0.0
        best_pretrain_state = None
        
        samples_seen = 0 

        for e in range(1, pretrain_epochs+1):
            running_loss = 0
            train_ita = 0

            num_batches = len(pretrain_loader)
            log_points = np.linspace(1, num_batches, 10, dtype=int)  # exactly 10 indices        
            
            for i, data in enumerate(pretrain_loader, 0):
                eeg_data, label = data
                eeg_data, label = eeg_data.to(device), label.to(device)
                optimizer.zero_grad()
                output = net(eeg_data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.to('cpu').item()
                train_ita = i

                batch_size_now = label.size(0)
                samples_seen += batch_size_now

                if (i+1) in log_points:
                    net.eval()
                    acc_step, class_acc_step = test_accuracy(net, pretrain_val_loader, device)
                    acc_test_step, _ = test_accuracy(net, pretrain_test_loader, device)
                    net.train()

                    with open(acc_over_samples_file, 'a') as f:
                        f.write(f"pretrain,{e},{i+1},{samples_seen},{acc_step:.6f},{class_acc_step[0]:.6f},{class_acc_step[1]:.6f},{acc_test_step:.6f}\n")

            # Clamp decay parameters
            for param in decay_params:
                param.data = param.data.clamp(min=1e-7)

            # Evaluate on 20% of data
            net.eval()
            acc_val, class_acc_val = test_accuracy(net, pretrain_val_loader, device)
            acc_test, _ = test_accuracy(net, pretrain_test_loader, device)
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
                f.write(f"pretrain,{e},{avg_loss:.6f},{acc_val:.6f},{class_acc_val[0]:.6f},{class_acc_val[1]:.6f},{acc_test:.6f},{current_lr:.8f}\n")

            scheduler.step()

            # ---- Save the best model ----
            # if acc_val > best_val_acc:
            if e == pretrain_epochs:
                best_val_acc = acc_val
                best_pretrain_state = copy.deepcopy(net.state_dict())

            # Save parameters after every 3 finetuning epoch OR if it's the final epoch
            save_parameters_to_h5(net, h5_filepath, epoch=e, stage="pretrain", save_interval=1)

            
        # Restore best pretrained model
        if best_pretrain_state:
            net.load_state_dict(best_pretrain_state)
        
        # Save pre-trained model
        if best_pretrain_state:
            pretrain_weights_path = os.path.join(save_dir, f"snn_pretrained_model_with_non_selected_participants_seed_{seed}_fold_{fold}_{timestamp}.pth")
            torch.save(best_pretrain_state, pretrain_weights_path)
            print(f"Pre-trained model saved to {pretrain_weights_path}")

        pretrain_time = time.time() - pretrain_start_time
        print(f"Pre-training Time: {pretrain_time:.2f} seconds ({pretrain_time/60:.2f} minutes)")

            
        # Save timing information to H5 file
        with h5py.File(h5_filepath, 'a') as f:
            timing_group = f.create_group('transfer_learning_metrics')
            timing_group.create_dataset('pretrain_time', data=pretrain_time)
            timing_group.attrs['device'] = str(device)
            timing_group.attrs['pretraining_batch_size'] = pretraining_batch_size
            timing_group.attrs['pretrain_epochs'] = pretrain_epochs

    
    return net, h5_filepath


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
    
    print("Starting SNN pretraining with parameter freezing for transfer learning")        

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    save_dir = f"snn_experiment_results/pretrained_model_5_fold"

    net, h5_file = pretrain_network_transfer_learning(
        dataset=DATASET,
        dataset_kwargs=ds_params,
        spike_ts=SPIKE_TS,
        param_list=PARAM_LIST,
        pretraining_batch_size=args.pretraining_batch_size,
        pretrain_epochs=args.pretrain_epochs,
        lr=[args.lr, args.lr, args.lr],
        weight_decays=[args.neuron_decay, args.ts_decay, args.wt_decay],
        save_dir=save_dir,
        seed = args.seed,
        args=args
    )

    print(f"\n=== TRAINING COMPLETED ===")
   
