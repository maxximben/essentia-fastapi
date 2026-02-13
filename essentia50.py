import json
import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

with open('models/msd-musicnn-1.json', 'r') as f:
    metadata = json.load(f)

tags = metadata['classes']
print(f"Всего тегов в модели: {len(tags)}")
print("Примеры тегов:", tags[:10], "...\n")

audio_path = "music/Never Gonna Give You Up - Rick Astley.m4a"
audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()

print(f"Аудио загружено: длина {len(audio)/16000:.1f} секунд\n")


print("Запуск автотеггинга...")
framewise_activations = TensorflowPredictMusiCNN(
    graphFilename="models/msd-musicnn-1.pb",
    output="model/Sigmoid"
)(audio)

global_probs = np.mean(framewise_activations, axis=0)

top_n = 10
top_indices = np.argsort(global_probs)[-top_n:][::-1]

print(f"Топ-{top_n} тегов для трека '{audio_path}':")
print("-" * 50)
for idx in top_indices:
    tag = tags[idx]
    prob = global_probs[idx]
    print(f"{tag:.<25} {prob:>6.3f}  ({prob*100:5.1f}%)")

print("-" * 50)