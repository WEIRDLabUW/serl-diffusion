import pickle
from typing import Union, List

import PIL.Image
import numpy as np
import open_clip
import torch
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
    pca_size: int = 52
    num_clusters: int = 1000
    clip_batch_size: int = 128


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="downample_dataset_cfg", node=DownsampleImageConfig)


@hydra.main(config_name="downample_dataset_cfg", version_base=None)
def main(cfg: DownsampleImageConfig):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k',
                                                                 device='cuda')
    data = load_data(cfg.dataset_path)
    right_im = [x['image_1'] for x in data]

    # # pick 100 random images
    # indexes = np.random.choice(len(right_im), 100, replace=False)
    # right_im = [right_im[i] for i in indexes]

    embeddings = get_embeddings(right_im, model, preprocess, cfg.clip_batch_size)
    start = time.time()
    embeddings = dimentionality_reduction(embeddings, num_components=cfg.pca_size)
    embeddings = np.stack(embeddings)
    labels, centers = cluster_with_kmeans(embeddings, num_clusters=cfg.num_clusters)
    closest_mean_point_indexes = get_closest_mean_point_indexes(embeddings, centers)
    print(time.time() - start)
    np.save('labels.npy', closest_mean_point_indexes)


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


def dimentionality_reduction(embeddings: List[torch.Tensor], num_components: int = 10) -> List[np.ndarray]:
    embeddings = np.stack([x.numpy() for x in embeddings])
    pca = PCA(n_components=num_components)
    embeddings = pca.fit_transform(embeddings)
    return embeddings


def cluster_with_kmeans(embeddings: np.ndarray, num_clusters: int = 10) -> np.ndarray:
    clasifier = KMeans(n_clusters=num_clusters, init='k-means++')
    clasifier.fit(embeddings)
    assert clasifier.n_iter_ < clasifier.max_iter
    return clasifier.labels_, clasifier.cluster_centers_


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
    return loaded_data


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
