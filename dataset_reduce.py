import copy
import os
import pickle
from typing import Union, List, Tuple
import math

import cv2
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import open_clip
import torch
from video_recorder import VideoRecorder

from PIL import Image
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
import time
import hydra
from dataclasses import dataclass


# Note hardcoded 'wrist_1' key for the image space. It only uses one camera for the image space.
@dataclass
class DownsampleImageConfig:
    dataset_path: str = "${hydra:runtime.cwd}/peg_insert_100_demos_2024-02-11_13-35-54.pkl"
    pca_size: int = -1
    num_clusters: int = 6
    clip_batch_size: int = 256
    save_path: str = "${hydra:runtime.cwd}/drawer_cluster/6Clusters"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="downample_dataset_cfg", node=DownsampleImageConfig)


@hydra.main(config_name="downample_dataset_cfg", version_base=None)
def main(cfg: DownsampleImageConfig):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k',
                                                                 device='cuda')
    model.eval()
    # data = load_data(cfg.dataset_path)
    data = load_video('outputs/dataset_video.mp4')
    right_im = [x['image_1'] for x in data]
    os.makedirs(cfg.save_path, exist_ok=True)
    # # pick 100 random images
    # indexes = np.random.choice(len(right_im), 100, replace=False)
    # right_im = [right_im[i] for i in indexes]

    embeddings = get_embeddings(copy.deepcopy(right_im), model, preprocess, cfg.clip_batch_size)
    print('got embeddings')
    start = time.time()
    if cfg.pca_size > 0:
        embeddings = dimensionality_reduction(embeddings, num_components=cfg.pca_size)
    print('pca done')
    embeddings = np.stack(embeddings)
    labels, centers = cluster_with_kmeans(embeddings, num_clusters=cfg.num_clusters)
    print('clustered')
    print(labels)
    closest_mean_point_indexes = get_closest_mean_point_indexes(embeddings, centers)
    print(time.time() - start)
    for idx, i in enumerate(closest_mean_point_indexes):
        cluster = labels[i]
        num_in_cluster = np.sum(labels == cluster)
        plt.imshow(right_im[i])
        plt.savefig(f'{cfg.save_path}/{num_in_cluster}_points_cluster_{idx}.png')

    np.save(f'{cfg.save_path}/new_labels.npy', closest_mean_point_indexes)


def get_embeddings(images: Union[List[PIL.Image.Image], PIL.Image.Image], model: open_clip.CLIP, preprocess,
                   batch_size: int = 32) -> List[torch.Tensor]:
    if isinstance(images, PIL.Image.Image):
        images = [images]

    for i in range(len(images)):
        images[i] = preprocess(images[i])

    num_batches = len(images) // batch_size
    if len(images) % batch_size != 0:
        num_batches += 1

    embeddings = []
    for batch in range(num_batches):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(images))
        input = torch.stack(images[start:end]).to('cuda')
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(input)

        for i in range(len(image_features)):
            embeddings.append(image_features[i].cpu())

    return embeddings


def dimensionality_reduction(embeddings: List[torch.Tensor], num_components: int = 10) -> List[np.ndarray]:
    embeddings = np.stack([x.numpy() for x in embeddings])
    pca = PCA(n_components=num_components)
    embeddings = pca.fit_transform(embeddings)
    return embeddings


def cluster_with_kmeans(embeddings: np.ndarray, num_clusters: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    clasifier = KMeans(n_clusters=num_clusters, init='k-means++')
    clasifier.fit(embeddings)
    assert clasifier.n_iter_ < clasifier.max_iter
    return clasifier.labels_, clasifier.cluster_centers_


def save_video(frames, path, fps):
    recorder = VideoRecorder(".", render_size=256, fps=fps)
    for i in range(len(frames)):
        if i == 0:
            recorder.init(np.array(frames[i]))
        else:
            recorder.record(np.array(frames[i]))
    recorder.save(f'{path}')


def get_closest_mean_point_indexes(embeddings: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = np.linalg.norm(embeddings[:, np.newaxis, :] - centers, axis=2)
    return np.argmin(diff, axis=0)


def load_data(dataset_path: str) -> List[dict]:
    data_dict = pickle.load(open(dataset_path, 'rb'))
    loaded_data = []
    for i in range(len(data_dict)):
        obs = data_dict[i]['observations']
        loaded_data.append({
            'image_1': Image.fromarray(obs['wrist_1']),
            'image_2': Image.fromarray(obs['wrist_2']),
        })
        # if data_dict[i]['dones']:
        #     break
    return loaded_data


def load_video(video_path: str):
    vid = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append({'image_1': Image.fromarray(frame)})
    return frames



if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
