import json
import faiss
from deepface import DeepFace

from utils import *
from config import *


#### ЕТАП І - створення векторної бази даних ####
df = DeepFace.find(img_path=r"C:\Users\HP\Downloads\adam_brody.jpg",
                   db_path=image_database_path,
                   model_name='Facenet512',
                   enforce_detection=False,
                   detector_backend='mtcnn')

# Репрезентації автоматично зберігаються в папці з зображеннями
# Ембединги переносимо в загальну папку з ресурсами

#### ЕТАП ІI - створення векторної бази даних ####
emb_dict = load_representations(embeddings_database_path)

genuine_map, impostor_map = create_genuine_impostor_maps(embeddings_dictionary=embeddings_database_path,
                                                         max_pairs=120)

thresholds: Dict[str, float] = {}

for actor in genuine_map:
    gen_scores = [cosine_similarity(emb_dict[a], emb_dict[b])
                  for a, b in genuine_map[actor]]
    imp_scores = [cosine_similarity(emb_dict[a], emb_dict[b])
                  for a, b in impostor_map[actor]]
    if not gen_scores or not imp_scores:
        print(f"Skipping actor {actor}: insufficient data.")
        continue
    threshold = calibrate_threshold(gen_scores, imp_scores)
    thresholds[actor] = threshold

with open(actor_distance_thresholds_path, 'w') as f:
    json.dump(thresholds, f, indent=2)

#### ЕТАП ІII - обчислення субцентроїдів та створення FAISS індексу ####
subcentroids, actor_map = calculate_subcentroids(embeddings_database_path)

M = len(subcentroids)
if M == 0:
    raise ValueError('No subcentroids generated; check DBSCAN parameters or embeddings data')
D = subcentroids[0].shape[0]
data = np.stack(subcentroids, axis=0)

index = faiss.IndexFlatIP(D)

index.add(data)

faiss.write_index(index, index_path)
with open(actor_map_path, 'wb') as f:
    pickle.dump(actor_map, f)

# Словник з українськими назвами акторів створимо та збережемо вручну