import torch
from torch.utils.data import Dataset
import numpy as np
from utility import get_eeg_2d_map, generate_left_right_dataset_from_saved_data,\
    generate_left_feet_classes_dataset_from_saved_data, generate_right_feet_classes_dataset_from_saved_data,\
    generate_left_right_rest_feet_dataset_from_saved_data, \
    generate_fist_feet_dataset_from_saved_data

class EEGDataset2DLeftRight(Dataset):
    """ Reguar Dataset for EEGMMIDB data """
    def __init__(self, base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                 use_imagery, transform=None):
        """
        :param base_route: base route to EEGMMIDB data
        :param subject_id_list: list of subjects
        :param start_ts: start timestep for each slice for epoch
        :param end_ts: maximum end timestep for each slice for epoch
        :param window_ts: window timestep for each epoch
        :param overlap_ts: overlap timestep between two window
        :param use_imagery: if true use imagery data instead of movement data
        :param transform: optional transform to be applied on a sample
        """
        self.data, self.label, self.epoch_ts, self.epoch_subjects = generate_left_right_dataset_from_saved_data(
            base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
            use_imagery, get_eeg_2d_map(), (10, 11)
        )

        self.transform = transform

    def __len__(self):
        """
        :return: length of the entire dataset
        """
        return self.data.shape[0]

    def __getitem__(self, item):
        """
        Get one entry of data by item
        :param item: index of data
        :return: data with label
        """
        item_label = self.label[item]
        item_data = self.data[item, :, :].reshape(10, 11, -1, 1) # -1  is automaticlly calculated time dimension and 1 is channel dimension

        if self.transform:
            item_data = self.transform(item_data)
        item_data_w_label = [item_data, item_label]
        return item_data_w_label

class EEGDataset2DFistFeet(Dataset):
    """ Reguar Dataset for EEGMMIDB data """
    def __init__(self, base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                 use_imagery, transform=None):
        """
        :param base_route: base route to EEGMMIDB data
        :param subject_id_list: list of subjects
        :param start_ts: start timestep for each slice for epoch
        :param end_ts: maximum end timestep for each slice for epoch
        :param window_ts: window timestep for each epoch
        :param overlap_ts: overlap timestep between two window
        :param use_imagery: if true use imagery data instead of movement data
        :param transform: optional transform to be applied on a sample
        """
        self.data, self.label, self.epoch_ts, self.epoch_subjects = generate_fist_feet_dataset_from_saved_data(
            base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
            use_imagery, get_eeg_2d_map(), (10, 11)
        )

        self.transform = transform

    def __len__(self):
        """
        :return: length of the entire dataset
        """
        return self.data.shape[0]

    def __getitem__(self, item):
        """
        Get one entry of data by item
        :param item: index of data
        :return: data with label
        """
        item_label = self.label[item]
        item_data = self.data[item, :, :].reshape(10, 11, -1, 1) # -1  is automaticlly calculated time dimension and 1 is channel dimension

        if self.transform:
            item_data = self.transform(item_data)
        item_data_w_label = [item_data, item_label]
        return item_data_w_label

class EEGDataset2DLeftRightRest(Dataset):
    """ Reguar Dataset for EEGMMIDB data """
    def __init__(self, base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                 use_imagery, transform=None, use_delta=False):
        """
        :param base_route: base route to EEGMMIDB data
        :param subject_id_list: list of subjects
        :param start_ts: start timestep for each slice for epoch
        :param end_ts: maximum end timestep for each slice for epoch
        :param window_ts: window timestep for each epoch
        :param overlap_ts: overlap timestep between two window
        :param use_imagery: if true use imagery data instead of movement data
        :param transform: optional transform to be applied on a sample
        """
        self.data, self.label, self.epoch_ts, self.epoch_subjects = generate_left_right_rest_dataset_from_saved_data(
            base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
            use_imagery, get_eeg_2d_map(), (10, 11)
        )

        self.transform = transform
        self.use_delta = use_delta

    def __len__(self):
        """
        :return: length of the entire dataset
        """
        return self.data.shape[0]

    def __getitem__(self, item):
        """
        Get one entry of data by item
        :param item: index of data
        :return: data with label
        """
        item_label = self.label[item]
        item_data = self.data[item, :, :].reshape(10, 11, -1, 1) # -1  is automaticlly calculated time dimension and 1 is channel dimension

        # print(f"Origianl range: [{item_data.min():.4f}, {item_data.max():.4f}]")
        # print(f"Original mean: {item_data.mean():.4f}, std: {item_data.std():.4f}")

        if self.use_delta:
            delta = np.diff(item_data, axis=2, prepend=item_data[:, :, [0], :])
            #item_data = np.concatenate([item_data, delta], axis=3)  # (H, W, T, C=2)
            item_data = delta # Delta encoding only, channel=1
            # print(f"Delta range: [{item_data.min():.4f}, {item_data.max():.4f}]")
            # print(f"Delta mean: {item_data.mean():.4f}, std: {item_data.std():.4f}")

            

        if self.transform:
            item_data = self.transform(item_data)
        item_data_w_label = [item_data, item_label]
        return item_data_w_label


class EEGDatasetLeftFeet(Dataset):
    """ Reguar Dataset for EEGMMIDB data """
    def __init__(self, base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                 use_imagery, transform=None):
        """
        :param base_route: base route to EEGMMIDB data
        :param subject_id_list: list of subjects
        :param start_ts: start timestep for each slice for epoch
        :param end_ts: maximum end timestep for each slice for epoch
        :param window_ts: window timestep for each epoch
        :param overlap_ts: overlap timestep between two window
        :param use_imagery: if true use imagery data instead of movement data
        :param use_no_movement: if true use no movement as class 0 in movement data
        :param transform: optional transform to be applied on a sample
        """
        self.data, self.label, self.epoch_ts, self.epoch_subjects = generate_left_feet_classes_dataset_from_saved_data(
            base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
            use_imagery, get_eeg_2d_map(), (10, 11)
        )
        self.transform = transform

    def __len__(self):
        """
        :return: length of the entire dataset
        """
        return self.data.shape[0]

    def __getitem__(self, item):
        """
        Get one entry of data by item
        :param item: index of data
        :return: data with label
        """
        item_label = self.label[item]
        item_data = self.data[item, :, :].reshape(10, 11, -1, 1)
        if self.transform:
            item_data = self.transform(item_data)
        item_data_w_label = [item_data, item_label]
        return item_data_w_label


class EEGDatasetRightFeet(Dataset):
    """ Reguar Dataset for EEGMMIDB data """
    def __init__(self, base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                 use_imagery, transform=None):
        """
        :param base_route: base route to EEGMMIDB data
        :param subject_id_list: list of subjects
        :param start_ts: start timestep for each slice for epoch
        :param end_ts: maximum end timestep for each slice for epoch
        :param window_ts: window timestep for each epoch
        :param overlap_ts: overlap timestep between two window
        :param use_imagery: if true use imagery data instead of movement data
        :param transform: optional transform to be applied on a sample
        """
        self.data, self.label, self.epoch_ts, self.epoch_subjects = generate_right_feet_classes_dataset_from_saved_data(
            base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
            use_imagery, get_eeg_2d_map(), (10, 11)
        )
        self.transform = transform

    def __len__(self):
        """
        :return: length of the entire dataset
        """
        return self.data.shape[0]

    def __getitem__(self, item):
        """
        Get one entry of data by item
        :param item: index of data
        :return: data with label
        """
        item_label = self.label[item]
        item_data = self.data[item, :, :].reshape(10, 11, -1, 1)
        if self.transform:
            item_data = self.transform(item_data)
        item_data_w_label = [item_data, item_label]
        return item_data_w_label


class EEGDataset2DLeftRightRestFeet(Dataset):
    """ Reguar Dataset for EEGMMIDB data """
    def __init__(self, base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                 use_imagery, transform=None):
        """
        :param base_route: base route to EEGMMIDB data
        :param subject_id_list: list of subjects
        :param start_ts: start timestep for each slice for epoch
        :param end_ts: maximum end timestep for each slice for epoch
        :param window_ts: window timestep for each epoch
        :param overlap_ts: overlap timestep between two window
        :param use_imagery: if true use imagery data instead of movement data
        :param transform: optional transform to be applied on a sample
        """
        self.data, self.label, self.epoch_ts, self.epoch_subjects = generate_left_right_rest_feet_dataset_from_saved_data(
            base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
            use_imagery, get_eeg_2d_map(), (10, 11)
        )

        self.transform = transform

    def __len__(self):
        """
        :return: length of the entire dataset
        """
        return self.data.shape[0]

    def __getitem__(self, item):
        """
        Get one entry of data by item
        :param item: index of data
        :return: data with label
        """
        item_label = self.label[item]
        item_data = self.data[item, :, :].reshape(10, 11, -1, 1)


        if self.transform:
            item_data = self.transform(item_data)
        item_data_w_label = [item_data, item_label]
        return item_data_w_label

class ToTensor(object):
    """ Transformation to convert ndarray to pytorch tensor"""
    def __call__(self, sample):
        """
        :param sample: ndarray
        :return: pytorch tensor
        """
        sample = sample.transpose((3, 0, 1, 2))
        sample = torch.from_numpy(sample).float()
        return sample