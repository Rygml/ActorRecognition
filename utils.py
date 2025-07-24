import os
import re
import random
import pickle
import numpy as np
from itertools import combinations
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple
from sklearn.metrics import roc_curve

def extract_actor_name(path: str) -> str:
    """
    Витягує ім'я актора без числових суфіксів із повного шляху.
    Наприклад: '.../Jake_Weber_48897_25986.jpeg' → 'Jake_Weber'
    """
    filename = os.path.basename(path)
    name, _ = os.path.splitext(filename)
    return re.sub(r'(_\d+)+$', '', name)


def load_representations(pkl_path: str) -> Dict[str, np.ndarray]:
    """Завантажує ембедінги з pickle у словник path->normalized vector."""
    with open(pkl_path, 'rb') as f:
        reps = pickle.load(f)
    emb_dict: Dict[str, np.ndarray] = {}
    for rep in reps:
        path = rep.get('identity')
        emb = rep.get('embedding')
        if path and emb is not None:
            arr = np.array(emb, dtype=np.float32)
            emb_dict[path] = arr / np.linalg.norm(arr)
    return emb_dict


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def calibrate_threshold(
    genuine: List[float],
    impostor: List[float],
    target_fpr: float = 0.01
) -> float:
    """Обчислює поріг за ROC, щоб FPR ≤ target_fpr."""
    y_true = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    y_score = np.concatenate([genuine, impostor])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    return float(thresholds[idx[-1]]) if idx.size > 0 else float(thresholds[0])

def create_genuine_impostor_maps(embeddings_dictionary, max_pairs=120):

    # Групування шляхів за іменем актора
    groups: Dict[str, List[str]] = {}
    for path in embeddings_dictionary.keys():
        actor = extract_actor_name(path)
        groups.setdefault(actor, []).append(path)

    # Genuine-пари: комбінації без повторень
    genuine_map: Dict[str, List[Tuple[str, str]]] = {}
    for actor, imgs in groups.items():
        pairs = list(combinations(imgs, 2))
        genuine_map[actor] = random.sample(pairs, max_pairs) if len(pairs) > max_pairs else pairs

    # Impostor-пари: випадкова пара різних акторів
    impostor_map: Dict[str, List[Tuple[str, str]]] = {}
    actors = list(groups.keys())
    for actor, imgs in groups.items():
        pairs: List[Tuple[str, str]] = []
        while len(pairs) < len(genuine_map.get(actor, [])):
            other = random.choice([a for a in actors if a != actor])
            img1 = random.choice(imgs)
            img2 = random.choice(groups[other])
            pairs.append((img1, img2))
        impostor_map[actor] = pairs

    return genuine_map, impostor_map

def calculate_subcentroids(embeddings_path, eps=0.5, min_samples=5):

    # Завантажити репрезентації
    with open(embeddings_path, 'rb') as f:
        reps = pickle.load(f)

    # Групувати ембедінги по актору
    groups = {}
    for rep in reps:
        emb = rep.get('embedding')
        if emb is None:
            continue
        actor = extract_actor_name(rep['identity'])
        groups.setdefault(actor, []).append(np.array(emb, dtype=np.float32))

    subcentroids = []
    actor_map = []  # паралельний список: індекс субцентроїда → ім'я актора

    # Кластеризація для кожного актора
    for actor, vecs in groups.items():
        X = np.stack(vecs, axis=0)
        # нормалізувати для cosine DBSCAN
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

        # DBSCAN з метрикою cosine
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = db.fit_predict(X_norm)

        unique_labels = set(labels)
        # обробка шуму та пустих кластерів
        if len(unique_labels - {-1}) == 0:
            # якщо жоден кластер не знайдено — беремо загальний центроїд
            centers = np.mean(X_norm, axis=0, keepdims=True)
            clusters = [centers]
        else:
            clusters = []
            for lbl in unique_labels:
                if lbl == -1:
                    continue
                members = X_norm[labels == lbl]
                center = np.mean(members, axis=0)
                clusters.append(center)

        for c in clusters:
            # нормалізація центроїда
            c = c / np.linalg.norm(c)
            subcentroids.append(c.astype(np.float32))
            actor_map.append(actor)

    return subcentroids, actor_map