import cv2
import math
import json
import faiss
import random
import pickle
import numpy as np
import streamlit as st
from typing import Tuple, List
from collections import defaultdict
from deepface.modules import detection, representation

from config import *


# Функції для кнопок навігації
def decrease_time():
    st.session_state.current_timestamp = max(0.0, st.session_state.current_timestamp - 0.5)
    st.session_state.slider_key += 1  # Змінюємо ключ для перемальовування слайдера


def increase_time():
    if 'cap' in st.session_state and st.session_state.cap is not None:
        duration = st.session_state.cap.get(cv2.CAP_PROP_FRAME_COUNT) / st.session_state.cap.get(cv2.CAP_PROP_FPS)
        st.session_state.current_timestamp = min(float(duration), st.session_state.current_timestamp + 0.5)
        st.session_state.slider_key += 1  # Змінюємо ключ для перемальовування слайдера


# Функція для отримання кадру за часовим кодом
def get_frame_at_timestamp(cap, timestamp):
    """
    Перемотуємо відео по номеру кадру замість POS_MSEC.
    """
    if cap is None or not cap.isOpened():
        return None, False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    if not ret:
        return None, False
    return frame, True

# Функція для гарантії, що зображення має підтримуваний тип даних для OpenCV
def ensure_uint8(img):
    """Convert image to uint8 if it's not already in that format"""
    if img is None:
        return None

    if isinstance(img, np.ndarray):
        # Check if image is float64 (CV_64F) and convert if needed
        if img.dtype == np.float64:
            # Scale to 0-255 range if needed
            if np.max(img) <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        elif img.dtype != np.uint8:
            # Handle other types by converting to uint8
            img = img.astype(np.uint8)

    return img
@st.cache_resource
def load_recognition_resources(
        index_path: str,
        actors_map_path: str,
        thresholds_path: str,
        ukr_actor_names_path: str
):
    # FAISS-індекс
    index = faiss.read_index(index_path)

    # Мапа акторів
    with open(actors_map_path, 'rb') as f:
        actor_map = pickle.load(f)

    # Індивідуальні трешхолди
    with open(thresholds_path, 'r') as f:
        inference_thresholds = json.load(f)

    # Імена акторів українською
    with open(ukr_actor_names_path, 'r', encoding='utf-8') as f:
        ukr_actor_names = json.load(f)

    return index, actor_map, inference_thresholds, ukr_actor_names


def recognize_actors(
        img: np.ndarray,
        default_threshold: float = 0.4,
        ratio_threshold: float = 1.1,
        model_name: str = 'Facenet512',
        detector_backend: str = 'mtcnn',
        align: bool = True,
        normalization: str = 'base'
) -> List[Tuple[np.ndarray, str, float, dict]]:
    """
    1) Завантажує FAISS-індекс та actor_map
    2) Детектує всі обличчя у img і робить для кожного ембединг
    3) Шукає двох найближчих сусідів (для ratio-test)
    4) Для кожного обличчя повертає кортеж (face_img, actor, similarity, facial_area)
       або (face_img, "Unknown", similarity, facial_area), якщо поріг не пройдений.
    """
    # Переконаємось у правильному форматі зображення
    img = ensure_uint8(img)

    # Знаходимо всі лиця на зображенні
    faces = detection.extract_faces(
        img_path=img,
        detector_backend=detector_backend,
        enforce_detection=False,
        align=align
    )

    h, w = img.shape[:2]

    # Відфільтровуємо хибні детекції за розміром
    faces = [
        f for f in faces
            if f.get('face') is not None and f['face'].size > 0
            and f['face'].shape[0] > 0 and f['face'].shape[1] > 0
            and f['face'].shape[0] < h and f['face'].shape[1] < w
            ]

    if not faces:
        return None

    results = []

    for face_info in faces:
        # Витягуємо саме лице з зображення та його координати
        face_img = face_info['face']
        facial_area = face_info['facial_area']

        # Переконаємось у правильному форматі зображення
        face_img = ensure_uint8(face_img)

        # Отримуємо ембединг
        rep = representation.represent(
            img_path=face_img,
            model_name=model_name,
            enforce_detection=True,
            detector_backend='skip',
            align=align,
            normalization=normalization
        )
        q = np.array(rep[0]['embedding'], dtype=np.float32)
        q /= np.linalg.norm(q)
        q = q.reshape(1, -1)

        # Шукаємо 2 найближчих центроїди
        D, I = INDEX.search(q, 2)
        sim1, sim2 = float(D[0][0]), float(D[0][1])
        actor1 = ACTOR_MAP[int(I[0][0])]

        # Перевіряємо чи ембединг знаходиться близько до кількох центроїдів одразу
        if sim1 / (sim2 + 1e-8) < ratio_threshold:
            results.append((face_img, "Unknown", sim1, facial_area))
            continue

        # Перевіряємо чи косинусна подібність достатня з найближчим центроїдом
        thr = INFERENCE_THRESHOLDS.get(actor1, default_threshold)
        if sim1 < thr:
            results.append((face_img, "Unknown", sim1, facial_area))
            continue

        # Повертаємо інформацію про актора та обличчя в разі успіху
        results.append((face_img, actor1, sim1, facial_area))

    return results

def brush_stroke_frame(image, x, y, w, h, stroke_color=(41, 128, 185), stroke_width=None, num_strokes=4):
    """
    Створює мінімалістичну рамку з мазків пензля навколо обличчя.

    """

    # Додаємо відступ для рамки
    padding = int(min(w, h) * 0.2)

    # Координати рамки з урахуванням відступу
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(image.shape[1], x + w + padding), min(image.shape[0], y + h + padding)

    # Автоматичне визначення товщини мазка
    if stroke_width is None:
        stroke_width = max(3, int(min(w, h) / 25))

    # Обмежуємо кількість мазків від 1 до 4
    num_strokes = max(1, min(4, num_strokes))

    # Малюємо мазки пензля
    draw_brush_strokes(image, x1, y1, x2, y2, stroke_color, stroke_width, num_strokes)


def draw_brush_strokes(image, x1, y1, x2, y2, color, width, num_strokes):
    """
    Малює художні мазки пензля навколо вказаної області.

    Args:
        image: зображення для малювання
        x1, y1: верхній лівий кут області
        x2, y2: нижній правий кут області
        color: колір мазків
        width: товщина мазків
        num_strokes: кількість мазків (1-4)
    """
    # Сторони прямокутника
    sides = []

    # Верхня сторона
    if num_strokes >= 1:
        sides.append([(x1, y1), (x2, y1)])

    # Права сторона
    if num_strokes >= 2:
        sides.append([(x2, y1), (x2, y2)])

    # Нижня сторона
    if num_strokes >= 3:
        sides.append([(x2, y2), (x1, y2)])

    # Ліва сторона
    if num_strokes >= 4:
        sides.append([(x1, y2), (x1, y1)])

    # Малюємо кожен мазок пензля
    for start, end in sides:
        # Створюємо точки для кривої
        control_points = create_brush_curve(start, end)

        # Малюємо основну криву
        draw_smooth_curve(image, control_points, color, width)

        # Додаємо невеликі варіації для органічності мазка
        alpha = 120  # Прозорість додаткових мазків
        light_color = tuple(min(255, c + 40) for c in color)  # Світліший відтінок

        # Додаємо 1-2 додаткові тонкі мазки для ефекту
        for _ in range(random.randint(1, 2)):
            # Зміщуємо контрольні точки для варіації
            varied_points = [(p[0] + random.randint(-3, 3), p[1] + random.randint(-3, 3))
                             for p in control_points]

            # Малюємо додатковий мазок
            draw_smooth_curve(image, varied_points, light_color, max(1, width // 3), alpha)


def create_brush_curve(start, end):
    """
    Створює контрольні точки для мазка пензля між двома точками.

    Args:
        start: початкова точка (x, y)
        end: кінцева точка (x, y)

    Returns:
        list: список контрольних точок для кривої
    """
    # Визначаємо проміжні точки
    length = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    num_points = max(5, int(length / 20))

    # Створюємо базові контрольні точки
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = int(start[0] * (1 - t) + end[0] * t)
        y = int(start[1] * (1 - t) + end[1] * t)

        # Додаємо художню недосконалість (варіацію)
        max_deviation = int(length / 15)

        # На початку і кінці менше відхилення
        factor = 4 * t * (1 - t)  # максимум у центрі, мінімум на кінцях
        deviation_x = int(random.randint(-max_deviation, max_deviation) * factor)
        deviation_y = int(random.randint(-max_deviation, max_deviation) * factor)

        points.append((x + deviation_x, y + deviation_y))

    # Забезпечуємо, щоб початкова і кінцева точки відповідали
    points[0] = start
    points[-1] = end

    return points


def draw_smooth_curve(image, points, color, width, alpha=255):
    """
    Малює плавну криву по контрольних точках.

    Args:
        image: зображення для малювання
        points: список контрольних точок
        color: колір кривої
        width: товщина кривої
        alpha: прозорість (0-255)
    """
    # Створюємо шар для малювання з прозорістю
    overlay = image.copy()

    # Конвертуємо список точок у масив точок для OpenCV
    points_array = np.array(points, dtype=np.int32)

    # Малюємо плавну криву через точки
    for i in range(len(points) - 1):
        pt1 = points[i]
        pt2 = points[i + 1]

        # Варіюємо товщину для ефекту мазка пензля
        local_width = max(1, int(width * (0.8 + 0.4 * random.random())))

        cv2.line(overlay, pt1, pt2, color, local_width, cv2.LINE_AA)

    # Накладаємо шар з прозорістю
    cv2.addWeighted(overlay, alpha / 255, image, 1 - alpha / 255, 0, image)


DETECTION_THRESHOLD = 0.7
INDEX, ACTOR_MAP, INFERENCE_THRESHOLDS, UKR_ACTOR_NAMES = load_recognition_resources(
    index_path=index_path,
    actors_map_path=actor_map_path,
    thresholds_path=actor_distance_thresholds_path,
    ukr_actor_names_path=ukr_actor_names_path
)


def get_face_embeddings(
        img: np.ndarray,
        default_threshold: float = 0.4,
        ratio_threshold: float = 1.3,
        model_name: str = 'Facenet512',
        detector_backend: str = 'mtcnn',
        align: bool = True,
        normalization: str = 'base'
) -> List[Tuple[np.ndarray, str, float, dict]]:
    """
    1) Завантажує FAISS-індекс та actor_map
    2) Детектує всі обличчя у img і робить для кожного ембединг
    3) Шукає двох найближчих сусідів (для ratio-test)
    4) Для кожного обличчя повертає кортеж (face_img, actor, similarity, facial_area)
       або (face_img, "Unknown", similarity, facial_area), якщо поріг не пройдений.
    """
    # Переконаємось у правильному форматі зображення
    img = ensure_uint8(img)

    # Знаходимо всі лиця на зображенні
    faces = detection.extract_faces(
        img_path=img,
        detector_backend=detector_backend,
        enforce_detection=False,
        align=align
    )

    h, w = img.shape[:2]

    # Відфільтровуємо хибні детекції за розміром
    faces = [
        f for f in faces
            if f.get('face') is not None and f['face'].size > 0
            and f['face'].shape[0] > 0 and f['face'].shape[1] > 0
            and f['face'].shape[0] < h and f['face'].shape[1] < w
            ]

    results = []

    for face_info in faces:
        # Витягуємо саме лице з зображення та його координати
        face_img = face_info['face']

        # Переконаємось у правильному форматі зображення
        face_img = ensure_uint8(face_img)

        # Отримуємо ембединг
        rep = representation.represent(
            img_path=face_img,
            model_name=model_name,
            enforce_detection=True,
            detector_backend='skip',
            align=align,
            normalization=normalization
        )
        q = np.array(rep[0]['embedding'], dtype=np.float32)
        q /= np.linalg.norm(q)
        q = q.reshape(1, -1)

        # Шукаємо 2 найближчих центроїди
        D, I = INDEX.search(q, 2)
        sim1, sim2 = float(D[0][0]), float(D[0][1])
        actor1 = ACTOR_MAP[int(I[0][0])]

        # Перевіряємо чи ембединг знаходиться близько до кількох центроїдів одразу
        if sim1 / (sim2 + 1e-8) < ratio_threshold:
            results.append(("Unknown", q))
            continue

        # Перевіряємо чи косинусна подібність достатня з найближчим центроїдом
        thr = INFERENCE_THRESHOLDS.get(actor1, default_threshold)
        if sim1 < thr:
            results.append(("Unknown", q))
            continue

        # Повертаємо інформацію про актора та обличчя в разі успіху
        results.append((actor1, q))

    return results

def process_video_embeddings(
        video_path: str,
        frame_step_sec: float = 1.0
) -> List[Tuple[np.ndarray, str, float]]:
    """
    Обробляє відео по кадрах, отримуючи ембединги облич із часовими мітками.

    Parameters:
    - video_path: шлях до відео (тимчасовий файл після upload)
    - recognize_fn: функція розпізнавання, яка повертає: (img, name, sim, box, emb)
    - frame_step_sec: крок у секундах між кадрами

    Returns:
    - Список кортежів (embedding, actor_name, timestamp)
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Не вдалося відкрити відеофайл.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = int(fps * frame_step_sec)
    embeddings_data = []

    # st.info("🔄 Знаходимо та розпізнаємо обличчя")
    progress = st.progress(0)

    for frame_idx in range(0, total_frames, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = ensure_uint8(rgb_frame)

        try:
            results = get_face_embeddings(rgb_frame)  # (img, name, sim, box, emb)
            timestamp = frame_idx / fps

            for actor_name, emb in results:
                embeddings_data.append((emb, actor_name, timestamp))

        except Exception as e:
            st.warning(f"⚠️ Помилка на кадрі {frame_idx}: {str(e)}")
            continue

        # Оновлення прогрес-бару
        percent = frame_idx / total_frames
        progress.progress(min(int(percent * 100), 100))

    cap.release()
    progress.progress(100)
    return embeddings_data

def cosine_sim(a, b):
    a = a.ravel()
    b = b.ravel()
    return np.clip(np.dot(a, b), -1.0, 1.0)


def track_actors_online(frames_embeddings: List[Tuple[np.ndarray, str, float]]) -> List[dict]:
    """
    Онлайн-кластеризація ембедингів на основі cosine similarity
    frames_embeddings — список кортежів: (embedding, actor_name, timestamp)
    """
    known_clusters = []  # список словників: {id, embeddings, actor_names, timestamps}
    threshold = 0.75  # якщо cosine similarity > threshold — вважається тією ж особою
    next_cluster_id = 0

    for embedding, actor_name, timestamp in frames_embeddings:

        matched = False
        for cluster in known_clusters:
            similarities = [cosine_sim(embedding, e) for e in cluster['embeddings']]
            if max(similarities) > threshold:
                cluster['embeddings'].append(embedding)
                cluster['actor_names'].append(actor_name)
                cluster['timestamps'].append(timestamp)
                matched = True
                break

        if not matched:
            known_clusters.append({
                'id': next_cluster_id,
                'embeddings': [embedding],
                'actor_names': [actor_name],
                'timestamps': [timestamp]
            })
            next_cluster_id += 1

    # Формуємо результат
    results = []
    for cluster in known_clusters:
        names = [n for n in cluster['actor_names'] if n != 'Unknown']
        identity = (
            max(set(names), key=names.count) if names else f"Unknown #{cluster['id']}"
        )

        results.append({
            'identity': identity.replace("_", " "),
            'count': len(cluster['timestamps']),
            'timestamps': cluster['timestamps']
        })

    return results

def localize_identities(stats: List[dict], name_map: dict) -> List[dict]:
    """
    Заміна англійських імен на українські, і перетворення Unknown → Невідома особа #N (із перенумерацією).
    """
    localized = []
    unknown_counter = 1

    for entry in stats:
        identity = entry["identity"]

        # Якщо є переклад у словнику
        if identity in name_map:
            localized_name = name_map[identity]
        elif identity.lower().startswith("unknown"):
            localized_name = f"Невідома особа #{unknown_counter}"
            unknown_counter += 1
        else:
            localized_name = identity  # залишаємо як є

        localized.append({
            "identity": localized_name,
            "count": entry["count"],
            "timestamps": entry.get("timestamps", [])
        })

    return localized


def merge_duplicate_identities(stats: List[dict]) -> List[dict]:
    """
    Об’єднує кластери з однаковим іменем, сумуючи кількість появ.
    """
    merged = defaultdict(lambda: {"count": 0, "timestamps": []})
    for entry in stats:
        identity = entry["identity"]
        merged[identity]["count"] += entry["count"]
        merged[identity]["timestamps"].extend(entry.get("timestamps", []))

    return [
        {"identity": identity, "count": data["count"], "timestamps": data["timestamps"]}
        for identity, data in merged.items()
    ]

def sort_by_count_then_identity(entry):
    count = entry['count']
    is_unknown = entry['identity'].startswith("Невідома особа")
    return (-count, is_unknown)  # спочатку за count ↓, потім невідомі — внизу


def generate_statistics_once():
    if not st.session_state.stats_ready:
        with st.spinner("🕵️‍♂️ Знаходимо та розпізнаємо обличчя..."):
            embeddings_data = process_video_embeddings(
                video_path=st.session_state.temp_file_path,
                frame_step_sec=1.0
            )
            if not embeddings_data:
                st.warning("❌ Не вдалося витягти ембединги з відео.")
                return

            final_stats = track_actors_online(embeddings_data)
            if not final_stats:
                st.warning("😐 Обличчя не знайдено або не вдалося кластеризувати.")
                return

            merged_stats_en = merge_duplicate_identities(final_stats)
            merged_stats = localize_identities(merged_stats_en, UKR_ACTOR_NAMES)
            merged_stats.sort(key=sort_by_count_then_identity)

            st.session_state.stats_ready = True
            st.session_state.final_stats = merged_stats

def reset_state():
    st.session_state.video_file = None
    st.session_state.temp_file_path = None
    st.session_state.cap = None
    st.session_state.frame = None
    st.session_state.frame_image = None
    st.session_state.recognition_results = None
    st.session_state.detector_loaded = False
    st.session_state.current_timestamp = 0.0
    st.session_state.slider_key = 0
    st.session_state.stats_ready = False
    st.session_state.final_stats = None
