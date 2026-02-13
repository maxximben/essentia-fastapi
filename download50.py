# Теперь это модуль, а не скрипт. Импортируйте и вызывайте analyze_musicnn_from_url(url)
import json
import numpy as np
import requests
import tempfile
import os
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN


def download_audio_from_url(url, timeout=30):
    """Скачивает аудиофайл по прямой ссылке и возвращает путь к временному файлу"""
    print(f"Скачивание аудио из: {url}")

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        ext = '.mp3'
        if '.' in url.split('/')[-1]:
            ext = os.path.splitext(url.split('/')[-1])[1]

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)

            tmp_path = tmp_file.name

        print(f"Аудио скачано во временный файл: {tmp_path}")
        return tmp_path

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Не удалось скачать аудиофайл: {e}")


def analyze_musicnn_from_url(audio_url: str) -> dict:
    try:
        audio_path = download_audio_from_url(audio_url)
        with open('models/msd-musicnn-1.json', 'r') as f:
            metadata = json.load(f)
        tags = metadata['classes']

        audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()

        framewise_activations = TensorflowPredictMusiCNN(
            graphFilename="models/msd-musicnn-1.pb",
            output="model/Sigmoid"
        )(audio)

        global_probs = np.mean(framewise_activations, axis=0)

        top_n = 10
        top_indices = np.argsort(global_probs)[-top_n:][::-1]

        results = []
        for idx in top_indices:
            tag = tags[idx]
            prob = float(global_probs[idx])
            results.append({"tag": tag, "prob": prob, "percent": prob * 100})

        os.unlink(audio_path)
        return {
            "model": "musicnn",
            "total_tags": len(tags),
            "top_n": top_n,
            "tags": results
        }
    except Exception as e:
        raise RuntimeError(f"Ошибка в анализе MusiCNN: {e}")


# Для теста как скрипт (опционально)
if __name__ == "__main__":
    url = "https://s3.twcstorage.ru/dff2fb2a-f4c1-4ba0-a0ef-42aab0ae6870/audio/1770212025622_Never%20Gonna%20Give%20You%20Up%20-%20Rick%20Astley.mp3"
    print(json.dumps(analyze_musicnn_from_url(url), indent=2))