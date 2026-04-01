from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import json
import numpy as np
import requests
import tempfile
import os
import httpx
from typing import Dict, Any

from essentia.standard import (
    MonoLoader,
    TensorflowPredictMusiCNN,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredict2D,
)

app = FastAPI(title="Music Analysis Service")

musicnn_model = None
musicnn_tags = None

discogs_embedding_model = None
discogs_classifier_model = None
discogs_tags = None


@app.on_event("startup")
def load_models():
    global musicnn_model, musicnn_tags
    global discogs_embedding_model, discogs_classifier_model, discogs_tags

    print("🚀 Загрузка моделей...")

    with open("models/msd-musicnn-1.json", "r") as f:
        metadata = json.load(f)
    musicnn_tags = metadata["classes"]
    musicnn_model = TensorflowPredictMusiCNN(
        graphFilename="models/msd-musicnn-1.pb",
        output="model/Sigmoid"
    )

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


def download_audio_from_url(url: str, timeout: int = 45) -> str:
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
    framewise = musicnn_model(audio)
    global_probs = np.mean(framewise, axis=0)
    tag_probs = {musicnn_tags[i]: float(global_probs[i]) for i in range(len(musicnn_tags))}

    return {
        "model": "musicnn-msd",
        "total_tags": len(musicnn_tags),
        "tags": tag_probs,
    }


def analyze_discogs(audio: np.ndarray) -> Dict[str, Any]:
    embeddings = discogs_embedding_model(audio)
    predictions = discogs_classifier_model(embeddings)
    global_probs = np.mean(predictions, axis=0)

    tag_probs = {discogs_tags[i]: float(global_probs[i]) for i in range(len(discogs_tags))}

    return {
        "model": "discogs-effnet-400",
        "total_tags": len(discogs_tags),
        "tags": tag_probs,
    }


class AnalyzeRequest(BaseModel):
    job_id: str
    audio_url: str
    callback_url: str


@app.post("/analyze")
async def start_analysis(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        process_audio_with_callback,
        request.audio_url,
        request.job_id,
        request.callback_url,
    )

    return {
        "status": "accepted",
        "job_id": request.job_id,
        "message": "Анализ запущен. Результат будет отправлен на callback_url",
    }


async def process_audio_with_callback(audio_url: str, job_id: str, callback_url: str):
    audio_path = None
    try:
        audio_path = download_audio_from_url(audio_url)

        audio = MonoLoader(
            filename=audio_path,
            sampleRate=16000,
            resampleQuality=4,
        )()
        audio_length = len(audio) / 16000.0

        musicnn_results = analyze_musicnn(audio)
        discogs_results = analyze_discogs(audio)

        payload = {
            "job_id": job_id,
            "status": "completed",
            "audio_length_seconds": round(audio_length, 1),
            "musicnn": musicnn_results,
            "discogs": discogs_results,
            "error": None,
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.post(callback_url, json=payload)
            response.raise_for_status()

        print(f"✅ Webhook успешно отправлен (job {job_id})")

    except Exception as e:
        print(f"❌ Ошибка в обработке job {job_id}: {repr(e)}")

        error_payload = {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.post(callback_url, json=error_payload)
                response.raise_for_status()

            print(f"⚠️ Отправлена ошибка по webhook (job {job_id})")
        except Exception as webhook_error:
            print(f"❌ Не удалось отправить даже ошибку по webhook: {repr(webhook_error)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)