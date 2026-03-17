# from fastapi import FastAPI, Query
# import json
# import numpy as np
# import requests
# import tempfile
# import os
# from essentia.standard import MonoLoader, TensorflowPredictMusiCNN, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
# from typing import Dict, Any

# app = FastAPI()


# def download_audio_from_url(url: str, timeout: int = 45) -> str:
#     """Скачивает аудиофайл по URL и возвращает путь к временному файлу."""
#     print(f"Скачивание аудио: {url}")
#     try:
#         response = requests.get(url, stream=True, timeout=timeout)
#         response.raise_for_status()
#         ext = '.mp3'  # По умолчанию, для примера в URL
#         filename = url.split('/')[-1]
#         if '.' in filename:
#             ext = os.path.splitext(filename)[1] or '.mp3'
#         with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 if chunk:
#                     tmp_file.write(chunk)
#             tmp_path = tmp_file.name
#         print(f"Файл скачан: {tmp_path} ({os.path.getsize(tmp_path) / 1024 / 1024:.1f} MB)")
#         return tmp_path
#     except requests.exceptions.RequestException as e:
#         raise RuntimeError(f"Ошибка при скачивании аудио: {e}")


# def analyze_musicnn(audio_path: str) -> Dict[str, Any]:
#     """Анализ MusiCNN — возвращаем ВСЕ теги"""
#     try:
#         with open('models/msd-musicnn-1.json', 'r') as f:
#             metadata = json.load(f)
#         tags = metadata['classes']

#         audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
#         framewise_activations = TensorflowPredictMusiCNN(
#             graphFilename="models/msd-musicnn-1.pb",
#             output="model/Sigmoid"
#         )(audio)
#         global_probs = np.mean(framewise_activations, axis=0)

#         # Словарь: тег → вероятность
#         tag_probs = {tags[i]: float(global_probs[i]) for i in range(len(tags))}

#         return {
#             "model": "musicnn-msd",
#             "total_tags": len(tags),
#             "tags": tag_probs
#         }
#     except Exception as e:
#         raise RuntimeError(f"Ошибка в анализе MusiCNN: {e}")


# def analyze_discogs(audio_path: str) -> Dict[str, Any]:
#     """Анализ Discogs-EffNet — возвращаем ВСЕ 400 тегов"""
#     embedding_pb = "models/discogs-effnet-bs64-1.pb"
#     classifier_pb = "models/genre_discogs400-discogs-effnet-1.pb"
#     classifier_json = "models/genre_discogs400-discogs-effnet-1.json"

#     for f in [embedding_pb, classifier_pb, classifier_json]:
#         if not os.path.exists(f):
#             raise FileNotFoundError(f"Файл модели '{f}' не найден!")

#     try:
#         with open(classifier_json, 'r') as f:
#             metadata = json.load(f)
#         tags = metadata['classes']

#         audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
#         embedding_model = TensorflowPredictEffnetDiscogs(
#             graphFilename=embedding_pb,
#             output="PartitionedCall:1"
#         )
#         embeddings = embedding_model(audio)

#         classifier = TensorflowPredict2D(
#             graphFilename=classifier_pb,
#             input="serving_default_model_Placeholder",
#             output="PartitionedCall"
#         )
#         predictions = classifier(embeddings)
#         global_probs = np.mean(predictions, axis=0)

#         # Словарь: тег → вероятность
#         tag_probs = {tags[i]: float(global_probs[i]) for i in range(len(tags))}

#         return {
#             "model": "discogs-effnet-400",
#             "total_tags": len(tags),
#             "tags": tag_probs
#         }
#     except Exception as e:
#         raise RuntimeError(f"Ошибка в анализе Discogs: {e}")


# @app.get("/analyze")
# async def analyze_audio(audio_url: str = Query(..., description="URL аудиофайла")):
#     audio_path = None
#     try:
#         audio_path = download_audio_from_url(audio_url)
#         audio = MonoLoader(filename=audio_path, sampleRate=16000, resampleQuality=4)()
#         audio_length = len(audio) / 16000.0

#         musicnn_results = analyze_musicnn(audio_path)
#         discogs_results = analyze_discogs(audio_path)

#         return {
#             "audio_length_seconds": round(audio_length, 1),
#             "musicnn": musicnn_results,
#             "discogs": discogs_results,
#             "error": None
#         }
#     except Exception as e:
#         return {
#             "audio_length_seconds": 0.0,
#             "musicnn": None,
#             "discogs": None,
#             "error": str(e)
#         }
#     finally:
#         if audio_path and os.path.exists(audio_path):
#             try:
#                 os.unlink(audio_path)
#             except:
#                 pass

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import json
import numpy as np
import requests
import tempfile
import os
import httpx
from uuid import uuid4
from typing import Dict, Any

from essentia.standard import (
    MonoLoader,
    TensorflowPredictMusiCNN,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredict2D,
)

app = FastAPI(title="Music Analysis Service")

# ==================== ГЛОБАЛЬНЫЕ МОДЕЛИ (загружаются один раз) ====================
musicnn_model = None
musicnn_tags = None

discogs_embedding_model = None
discogs_classifier_model = None
discogs_tags = None


@app.on_event("startup")
def load_models():
    """Загружаем все модели при старте приложения (один раз)"""
    global musicnn_model, musicnn_tags
    global discogs_embedding_model, discogs_classifier_model, discogs_tags

    print("🚀 Загрузка моделей...")

    # MusiCNN
    with open("models/msd-musicnn-1.json", "r") as f:
        metadata = json.load(f)
    musicnn_tags = metadata["classes"]
    musicnn_model = TensorflowPredictMusiCNN(
        graphFilename="models/msd-musicnn-1.pb",
        output="model/Sigmoid"
    )

    # Discogs-EffNet
    with open("models/genre_discogs400-discogs-effnet-1.json", "r") as f:
        metadata = json.load(f)
    discogs_tags = metadata["classes"]

    discogs_embedding_model = TensorflowPredictEffnetDiscogs(
        graphFilename="models/discogs-effnet-bs64-1.pb",
        output="PartitionedCall:1"
    )
    discogs_classifier_model = TensorflowPredict2D(
        graphFilename="models/genre_discogs400-discogs-effnet-1.pb",
        input="serving_default_model_Placeholder",
        output="PartitionedCall"
    )

    print("✅ Все модели успешно загружены!")


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
def download_audio_from_url(url: str, timeout: int = 45) -> str:
    """Скачивает аудио и возвращает путь к временному файлу"""
    print(f"📥 Скачивание: {url}")
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        ext = os.path.splitext(url.split("/")[-1])[1] or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            tmp_path = tmp.name

        print(f"✅ Скачано: {tmp_path} ({os.path.getsize(tmp_path) / 1024 / 1024:.1f} MB)")
        return tmp_path
    except Exception as e:
        raise RuntimeError(f"Ошибка скачивания: {e}")


def analyze_musicnn(audio: np.ndarray) -> Dict[str, Any]:
    """MusiCNN — возвращает все теги"""
    framewise = musicnn_model(audio)
    global_probs = np.mean(framewise, axis=0)
    tag_probs = {musicnn_tags[i]: float(global_probs[i]) for i in range(len(musicnn_tags))}

    return {
        "model": "musicnn-msd",
        "total_tags": len(musicnn_tags),
        "tags": tag_probs,
    }


def analyze_discogs(audio: np.ndarray) -> Dict[str, Any]:
    """Discogs-EffNet 400 тегов"""
    embeddings = discogs_embedding_model(audio)
    predictions = discogs_classifier_model(embeddings)
    global_probs = np.mean(predictions, axis=0)

    tag_probs = {discogs_tags[i]: float(global_probs[i]) for i in range(len(discogs_tags))}

    return {
        "model": "discogs-effnet-400",
        "total_tags": len(discogs_tags),
        "tags": tag_probs,
    }


# ==================== Pydantic модель запроса ====================
class AnalyzeRequest(BaseModel):
    audio_url: str
    callback_url: str


# ==================== ОСНОВНОЙ ENDPOINT ====================
@app.post("/analyze")
async def start_analysis(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Запускает анализ асинхронно и сразу возвращает job_id"""
    job_id = str(uuid4())

    background_tasks.add_task(
        process_audio_with_callback,
        request.audio_url,
        job_id,
        request.callback_url,
    )

    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Анализ запущен (~30–40 сек). Результат будет отправлен на callback_url",
    }


# ==================== ФОНОВАЯ ОБРАБОТКА ====================
async def process_audio_with_callback(audio_url: str, job_id: str, callback_url: str):
    audio_path = None
    try:
        # 1. Скачиваем файл
        audio_path = download_audio_from_url(audio_url)

        # 2. Загружаем аудио ОДИН раз
        audio = MonoLoader(
            filename=audio_path,
            sampleRate=16000,
            resampleQuality=4,
        )()
        audio_length = len(audio) / 16000.0

        # 3. Анализируем (модели уже загружены)
        musicnn_results = analyze_musicnn(audio)
        discogs_results = analyze_discogs(audio)

        # 4. Формируем результат
        payload = {
            "job_id": job_id,
            "status": "completed",
            "audio_length_seconds": round(audio_length, 1),
            "musicnn": musicnn_results,
            "discogs": discogs_results,
            "error": None,
        }

        # 5. Отправляем webhook
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.post(callback_url, json=payload)
            response.raise_for_status()

        print(f"✅ Webhook успешно отправлен (job {job_id})")

    except Exception as e:
        error_payload = {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                await client.post(callback_url, json=error_payload)
            print(f"⚠️ Отправлена ошибка по webhook (job {job_id})")
        except:
            print(f"❌ Не удалось отправить даже ошибку по webhook: {e}")

    finally:
        # Удаляем временный файл
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass


# ==================== Запуск ====================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
                
