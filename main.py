from fastapi import FastAPI, Query
import json
import numpy as np
import requests
import tempfile
import os
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
from typing import Dict, Any

app = FastAPI()


def download_audio_from_url(url: str, timeout: int = 45) -> str:
    """Скачивает аудиофайл по URL и возвращает путь к временному файлу."""
    print(f"Скачивание аудио: {url}")
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        ext = '.mp3'  # По умолчанию, для примера в URL
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


def analyze_musicnn(audio_path: str) -> Dict[str, Any]:
    """Анализ MusiCNN — возвращаем ВСЕ теги"""
    try:
        with open('models/msd-musicnn-1.json', 'r') as f:
            metadata = json.load(f)
        tags = metadata['classes']

        audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
        framewise_activations = TensorflowPredictMusiCNN(
            graphFilename="models/msd-musicnn-1.pb",
            output="model/Sigmoid"
        )(audio)
        global_probs = np.mean(framewise_activations, axis=0)

        # Словарь: тег → вероятность
        tag_probs = {tags[i]: float(global_probs[i]) for i in range(len(tags))}

        return {
            "model": "musicnn-msd",
            "total_tags": len(tags),
            "tags": tag_probs
        }
    except Exception as e:
        raise RuntimeError(f"Ошибка в анализе MusiCNN: {e}")


def analyze_discogs(audio_path: str) -> Dict[str, Any]:
    """Анализ Discogs-EffNet — возвращаем ВСЕ 400 тегов"""
    embedding_pb = "models/discogs-effnet-bs64-1.pb"
    classifier_pb = "models/genre_discogs400-discogs-effnet-1.pb"
    classifier_json = "models/genre_discogs400-discogs-effnet-1.json"

    for f in [embedding_pb, classifier_pb, classifier_json]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Файл модели '{f}' не найден!")

    try:
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

        # Словарь: тег → вероятность
        tag_probs = {tags[i]: float(global_probs[i]) for i in range(len(tags))}

        return {
            "model": "discogs-effnet-400",
            "total_tags": len(tags),
            "tags": tag_probs
        }
    except Exception as e:
        raise RuntimeError(f"Ошибка в анализе Discogs: {e}")


@app.get("/analyze")
async def analyze_audio(audio_url: str = Query(..., description="URL аудиофайла")):
    audio_path = None
    try:
        audio_path = download_audio_from_url(audio_url)
        audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
        audio_length = len(audio) / 16000.0

        musicnn_results = analyze_musicnn(audio_path)
        discogs_results = analyze_discogs(audio_path)

        return {
            "audio_length_seconds": round(audio_length, 1),
            "musicnn": musicnn_results,
            "discogs": discogs_results,
            "error": None
        }
    except Exception as e:
        return {
            "audio_length_seconds": 0.0,
            "musicnn": None,
            "discogs": None,
            "error": str(e)
        }
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass


# Существующие эндпоинты оставлены для совместимости
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}