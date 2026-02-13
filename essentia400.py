import json
import numpy as np
import os
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D

embedding_pb = "models/discogs-effnet-bs64-1.pb"
classifier_pb = "models/genre_discogs400-discogs-effnet-1.pb"
classifier_json = "models/genre_discogs400-discogs-effnet-1.json"

for f in [embedding_pb, classifier_pb, classifier_json]:
    if not os.path.exists(f):
        print(f"Ошибка: файл '{f}' не найден! Скачайте с https://essentia.upf.edu/models.html")
        exit(1)

with open(classifier_json, 'r') as f:
    metadata = json.load(f)
tags = metadata['classes']
print(f"Модель Discogs-EffNet: {len(tags)} жанров/стилей")
print("Примеры тегов:", tags[:8], "...\n")

audio_path = "music/Never Gonna Give You Up - Rick Astley.m4a"
if not os.path.exists(audio_path):
    print(f"Файл '{audio_path}' не найден!")
    exit(1)

audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
print(f"Аудио загружено: {len(audio)/16000:.1f} секунд, 16 kHz\n")

print("Извлечение эмбеддингов Discogs-EffNet...")
embedding_model = TensorflowPredictEffnetDiscogs(
    graphFilename=embedding_pb,
    output="PartitionedCall:1"
)
embeddings = embedding_model(audio)

print("Классификация по 400 жанрам...")
classifier = TensorflowPredict2D(
    graphFilename=classifier_pb,
    input="serving_default_model_Placeholder",
    output="PartitionedCall"
)
predictions = classifier(embeddings)

global_probs = np.mean(predictions, axis=0)

top_n = 15
top_indices = np.argsort(global_probs)[-top_n:][::-1]

print(f"\nТоп-{top_n} жанров/стилей по Discogs-EffNet для '{audio_path}':")
print("-" * 60)
for idx in top_indices:
    tag = tags[idx]
    prob = global_probs[idx]
    print(f"{tag:.<45} {prob:>6.3f}  ({prob*100:5.1f}%)")
print("-" * 60)