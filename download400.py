# Теперь это модуль. Импортируйте и вызывайте analyze_discogs_from_url(url)
import json
import numpy as np
import requests
import tempfile
import os
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D


def download_audio_from_url(url, timeout=45):
    """Скачивает аудиофайл по прямой ссылке и возвращает путь к временному файлу"""
    print(f"Скачивание аудио: {url}")

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        ext = '.mp3'
        filename = url.split('/')[-1]
        if '.' in filename:
            ext = os.path.splitext(filename)[1] or '.mp3'

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)

            tmp_path = tmp_file.name

        print(f"Файл скачан: {tmp_path} ({os.path.getsize(tmp_path) / 1024 / 1024:.1f} MB)")
        return tmp_path

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ошибка при скачивании аудио: {e}")


def analyze_discogs_from_url(audio_url: str) -> dict:
    embedding_pb = "models/discogs-effnet-bs64-1.pb"
    classifier_pb = "models/genre_discogs400-discogs-effnet-1.pb"
    classifier_json = "models/genre_discogs400-discogs-effnet-1.json"
    for f in [embedding_pb, classifier_pb, classifier_json]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Файл '{f}' не найден! Скачайте с https://essentia.upf.edu/models.html")

    try:
        audio_path = download_audio_from_url(audio_url)
        with open(classifier_json, 'r') as f:
            metadata = json.load(f)
        tags = metadata['classes']

        audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()

        embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename=embedding_pb,
            output="PartitionedCall:1"
        )
        embeddings = embedding_model(audio)

        classifier = TensorflowPredict2D(
            graphFilename=classifier_pb,
            input="serving_default_model_Placeholder",
            output="PartitionedCall"
        )
        predictions = classifier(embeddings)

        global_probs = np.mean(predictions, axis=0)

        top_n = 15
        top_indices = np.argsort(global_probs)[-top_n:][::-1]

        results = []
        for idx in top_indices:
            tag = tags[idx]
            prob = float(global_probs[idx])
            results.append({"tag": tag, "prob": prob, "percent": prob * 100})

        os.unlink(audio_path)
        return {
            "model": "discogs",
            "total_tags": len(tags),
            "top_n": top_n,
            "tags": results
        }
    except Exception as e:
        raise RuntimeError(f"Ошибка в анализе Discogs: {e}")


# Для теста как скрипт (опционально)
if __name__ == "__main__":
    url = "https://s3.twcstorage.ru/dff2fb2a-f4c1-4ba0-a0ef-42aab0ae6870/audio/1770212025622_Never%20Gonna%20Give%20You%20Up%20-%20Rick%20Astley.mp3"
    print(json.dumps(analyze_discogs_from_url(url), indent=2))