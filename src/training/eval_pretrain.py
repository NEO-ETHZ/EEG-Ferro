import os
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    _REPO_ROOT / "dataset",
    _REPO_ROOT / "configs",
    _REPO_ROOT / "src" / "models",
    _REPO_ROOT / "src" / "devices",
):
    sys.path.insert(0, str(_path))
from utility import samples_per_class, train_validate_split_4_fold_CV_transfer_learning
from dataset import ToTensor, EEGDataset2DLeftRight, EEGDatasetRightFeet, EEGDatasetLeftFeet
from torch.utils.data import Dataset, DataLoader, sampler
from snn import WrapCUBASpikingCNN
import torch
import numpy as np
from datetime import datetime
from config import parse_args
import random




def evaluate_pretrained_model(pretrained_model_path, dataset, dataset_kwargs=dict(), target_subject_id=1,
                               save_dir="pretrain_evaluate_result", seed=None, batch_size=64, device=None):
    """
    Evaluate a pretrained model on 3-fold CV of one subject.
    Saves per-fold accuracies to a log file.

    :param pretrained_model_path: path to pretrained weights
    :param dataset: EEG dataset object
    :param target_subject: subject ID to evaluate
    :param save_dir: directory to save log file
    :param seed: random seed for reproducibility
    :param batch_size: dataloader batch size
    :param device: torch.device
    :return: dict with results per fold
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

     # Setup Dataset and split for transfer learning
    ds = dataset(**dataset_kwargs)

    # --- Split dataset into 3 folds
    folds = train_validate_split_4_fold_CV_transfer_learning(ds, target_subject_id, seed=seed)

    # --- Init model
    SPIKE_TS = 160
    PARAM_LIST = [0.1, 0.1, 0.1, 0.3, 0.01, 0.1, 0.01]

    net = WrapCUBASpikingCNN(spike_ts=SPIKE_TS, device=device, param_list=PARAM_LIST)
    net.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    net.to(device)
    net.eval()

    # --- Prepare logging
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    accuracy_file = os.path.join(save_dir, f"pretrained_model_accuracy_seed_{seed}_{timestamp}.txt")

    results = {}
    with open(accuracy_file, "w") as f:
        f.write("Fold,Val_Accuracy,Class0_Accuracy,Class1_Accuracy\n")

        # --- Evaluate per fold
        for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):

            val_sampler = sampler.SubsetRandomSampler(val_idx)

            val_loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                               sampler=val_sampler, num_workers=4, pin_memory=True)

            class_correct = np.zeros(2)
            class_total = np.zeros(2)

            with torch.no_grad():
                for eeg_data, label in val_loader:
                    eeg_data, label = eeg_data.to(device), label.to(device)
                    output = net(eeg_data)
                    _, predicted = torch.max(output, 1)
                    correct = (predicted == label).cpu().numpy()

                    for i in range(label.size(0)):
                        la = int(label[i].cpu().item())
                        class_correct[la] += correct[i]
                        class_total[la] += 1

            fold_acc = class_correct.sum() / class_total.sum()
            class_acc = class_correct / class_total

            results[fold_idx] = {
                "overall_acc": fold_acc,
                "class0_acc": class_acc[0],
                "class1_acc": class_acc[1]
            }

            f.write(f"{fold_idx},{fold_acc:.4f},{class_acc[0]:.4f},{class_acc[1]:.4f}\n")


    return results


if __name__ == "__main__":

    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DATASET = EEGDataset2DLeftRight  # Only Hand (L) vs Hand (R)
    USE_IMAGERY = True

    target_subject_lists = [[i + 1 for i in range(21)],
                           [i + 1 for i in range(21, 42)],
                           [i + 1 for i in range (42, 63)],
                           [i + 1 for i in range (63, 83)],
                           [84, 85, 86, 87, 90, 91, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 105, 107, 108, 109]
                           ]
    
    pretrained_models = ['model_online_seed_5_fold_1_dut_dut2_thr_0p025_thrasym_1p0_Ascale_1p0_Aasym_1p0_rsdc2c_0.0_20251127_215126.pth',
                         'model_online_seed_5_fold_2_dut_dut2_thr_0p025_thrasym_1p0_Ascale_1p0_Aasym_1p0_rsdc2c_0.0_20251127_222122.pth',
                         'model_online_seed_5_fold_3_dut_dut2_thr_0p025_thrasym_1p0_Ascale_1p0_Aasym_1p0_rsdc2c_0.0_20251127_225119.pth',
                         'model_online_seed_5_fold_4_dut_dut2_thr_0p025_thrasym_1p0_Ascale_1p0_Aasym_1p0_rsdc2c_0.0_20251127_232115.pth',
                         'model_online_seed_5_fold_5_dut_dut2_thr_0p025_thrasym_1p0_Ascale_1p0_Aasym_1p0_rsdc2c_0.0_20251127_235115.pth'                         
                        ]




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
            save_dir = f"snn_experiment_results/full_online_pretrained_model_per_subject_evaluation/fold_{k+1}/subject_id_{target_subject_id}"

            pretrained_model_path = f"snn_experiment_results/SSTL_pretrained_model/{pretrained_model}"

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            results = evaluate_pretrained_model(
                dataset=DATASET,
                dataset_kwargs=ds_params,
                target_subject_id=target_subject_id,
                pretrained_model_path=pretrained_model_path,
                save_dir=save_dir,
                seed = args.seed,
                batch_size=1
            )


