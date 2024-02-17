import numpy as np
import torch
import os
import pickle


def create_sample_indices(
        episode_ends: np.ndarray, sequence_length: int,
        pad_before: int = 0, pad_after: int = 0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class SERLImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 num_trajectories: int = 100,
                 image_keys: list = ['wrist_1', 'wrist_2'],
                 state_key: str = 'state',
                 ):

        # load the demonstration dataset:
        data = pickle.load(open(dataset_path, 'rb'))

        actions = []
        images = []
        states = []
        episode_ends = []

        for i in range(len(data)):
            states.append(data[i]['observations'][state_key])
            imgs = []
            for key in image_keys:
                img = data[i]['observations'][key]
                img = np.moveaxis(img, -1, 0)
                imgs.append(img)
            image = np.stack(imgs, axis=0)
            images.append(image)
            actions.append(data[i]['actions'])
            if data[i]['dones']:
                episode_ends.append(i + 1)
                # Temp solution
                if len(episode_ends) >= num_trajectories:
                    break

        # sample num_trajectories number of trajectories
        # TODO: I do not think this works
        # if num_trajectories < len(episode_ends):
            
        #     start_idxs = [0] + episode_ends[:-1]
        #     #idxs = np.random.choice(range(len(episode_ends)), num_trajectories, replace=False)
        #     #idxs = np.sort(idxs)
        #     idxs = np.array([0, 1])

        #     episode_intervals = [(start_idxs[i], episode_ends[i]) for i in idxs]
        #     actions = np.concatenate([np.array(actions[start: end]) for start, end in episode_intervals], axis=0)
        #     states = np.concatenate([np.array(states[start: end]) for start, end in episode_intervals], axis=0)
        #     images = np.concatenate([np.array(images[start: end]) for start, end in episode_intervals], axis=0)
        #     episode_ends = np.array([ep_end - start for start, ep_end in episode_intervals])

        #     # They are in the form of length of each episode, so we want to sum them together
        #     episode_ends = np.cumsum(episode_ends)
        #     # for i in range(len(episode_ends)):
        #     #     if i > 0:
        #     #         episode_ends[i] += episode_ends[i - 1]
        # else:
        actions = np.array(actions).astype(np.float32)
        states = np.array(states).astype(np.float32)
        episode_ends = np.array(episode_ends)
        images = np.array(images).astype(np.float32)

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'state': states,
            'action': actions,
        }

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        normalized_train_data['image'] = images

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['image'] = nsample['image'][:self.obs_horizon, :]
        nsample['state'] = nsample['state'][:self.obs_horizon, :]
        return nsample


if __name__ == "__main__":
    dataset_path = '/home/siri/Desktop/insertion_replication/serl/peg_insert_100_demos_2024-02-11_13-35-54.pkl'
    dataset1 = SERLImageDataset(
        dataset_path=dataset_path,
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        num_trajectories=100,
    )
    dataset2 = SERLImageDataset(
        dataset_path=dataset_path,
        pred_horizon=26,
        obs_horizon=2,
        action_horizon=8,
        num_trajectories=2,
    )
    x = dataset1[0] 
    y = dataset2[0] 
    print('here')

