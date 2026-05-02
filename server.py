import io
import os
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
import soundfile as sf


app = FastAPI(title="NeuTTS Air")

backbone_device = os.getenv("NEUTTS_BACKBONE_DEVICE", "cpu")
default_ref_audio = Path("/app/samples/jo.wav")
default_ref_text = Path("/app/samples/jo.txt")

tts = None
tts_lock = threading.Lock()


def _get_tts():
    global tts
    if tts is None:
        with tts_lock:
            if tts is None:
                from neutts import NeuTTS

                tts = NeuTTS(
                    backbone_repo=os.getenv("NEUTTS_BACKBONE_REPO", "neuphonic/neutts-air"),
                    backbone_device=backbone_device,
                    codec_repo=os.getenv("NEUTTS_CODEC_REPO", "neuphonic/neucodec"),
                    codec_device=os.getenv("NEUTTS_CODEC_DEVICE", backbone_device),
                )

    return tts


def _generate_with_reference(text: str, ref_audio_path: Path, ref_text: str):
    model = _get_tts()
    ref_codes = model.encode_reference(str(ref_audio_path))
    return model.infer(text, ref_codes, ref_text)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True, "model_loaded": tts is not None}


@app.post("/tts")
async def tts_endpoint(
    text: str = Form(...),
    ref_audio: UploadFile | None = File(default=None),
    ref_text: str | None = Form(default=None),
):
    if (ref_audio is None) != (ref_text is None):
        raise HTTPException(
            status_code=400,
            detail="Provide both ref_audio and ref_text for voice cloning.",
        )

    if ref_audio is not None and ref_text is not None:
        suffix = Path(ref_audio.filename or "reference.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(await ref_audio.read())

        try:
            audio = _generate_with_reference(text, temp_path, ref_text)
        finally:
            await ref_audio.close()
            temp_path.unlink(missing_ok=True)
    else:
        audio = _generate_with_reference(
            text,
            default_ref_audio,
            default_ref_text.read_text(encoding="utf-8").strip(),
        )

    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")