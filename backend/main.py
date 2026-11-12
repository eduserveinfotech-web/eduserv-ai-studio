import os
import json
import tempfile
import urllib.request
import urllib.parse
import edge_tts
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from huggingface_hub import InferenceClient
from gradio_client import Client, handle_file
from google import genai

app = FastAPI(title="Eduserv AI Studio - Cloud Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# किसी भी अनपेक्षित एरर पर भी CORS हेडर सुरक्षित रखने के लिए
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"},
    )

HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

CHARACTER_PACKS = {
    "Gussail Chacha":  {"voice": "hi-IN-MadhurNeural", "rate": "+10%", "pitch": "-18Hz"},
    "Chulbuli Dadi":   {"voice": "hi-IN-SwaraNeural",  "rate": "+16%", "pitch": "+24Hz"},
    "Filmi Villain":   {"voice": "hi-IN-MadhurNeural", "rate": "-12%", "pitch": "-30Hz"},
    "Cartoon Bacha":   {"voice": "hi-IN-SwaraNeural",  "rate": "+22%", "pitch": "+42Hz"},
    "Darawna Bhoot":   {"voice": "hi-IN-MadhurNeural", "rate": "-22%", "pitch": "-38Hz"},
    "Shararti Dost":   {"voice": "hi-IN-MadhurNeural", "rate": "+18%", "pitch": "+16Hz"},
}

@app.get("/")
def home():
    return {"status": "online", "hf_token_loaded": bool(HF_TOKEN), "gemini_loaded": bool(GEMINI_KEY)}

@app.post("/api/plan")
async def plan_reel(idea: str = Form(...)):
    if not GEMINI_KEY:
        return {"dialogue": idea, "image_prompt": f"{idea}, funny 3D talking character", "emotion": "Comedy"}

    client = genai.Client(api_key=GEMINI_KEY)
    prompt = f"""
    You are a viral Hindi Reels creator. User Idea: "{idea}"
    Return ONLY valid JSON with 3 keys:
    1. "dialogue": Funny, expressive Hindi dialogue (20-35 words) with pauses '...' and '!'.
    2. "image_prompt": English prompt for a 3D Pixar-style character (vegetable, snake, animal) with big eyes and clear mouth facing camera.
    3. "emotion": One of ["Comedy", "Angry", "Dramatic", "Excited"].
    """
    try:
        res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        clean = res.text.replace("```json", "").replace("```", "").strip()
        return JSONResponse(content=json.loads(clean))
    except Exception:
        return {"dialogue": idea, "image_prompt": f"{idea}, cute 3D Pixar character", "emotion": "Comedy"}

@app.post("/api/image")
async def make_image(prompt: str = Form(...)):
    enhanced = (
        f"{prompt}, 3D Pixar animation style, big expressive cartoon eyes, "
        "clearly visible mouth, front-facing portrait, centered, studio lighting, 8k"
    )
    path = tempfile.mktemp(suffix=".png")

    # पहला प्रयास: Hugging Face Inference API
    if HF_TOKEN:
        try:
            client = InferenceClient(token=HF_TOKEN)
            img = client.text_to_image(enhanced, model="black-forest-labs/FLUX.1-schnell")
            img.save(path)
            return FileResponse(path, media_type="image/png")
        except Exception:
            pass

    # दूसरा पक्का और तेज़ प्रयास: Pollinations Flux Engine (बिना किसी टोकन लिमिट के)
    try:
        encoded_prompt = urllib.parse.quote(enhanced)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=768&height=1024&model=flux&nologo=true"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            with open(path, "wb") as f:
                f.write(resp.read())
        return FileResponse(path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Image generation failed: {str(e)}")

@app.post("/api/voice")
async def make_voice(text: str = Form(...), voice_pack: str = Form("Gussail Chacha")):
    pack = CHARACTER_PACKS.get(voice_pack, CHARACTER_PACKS["Gussail Chacha"])
    path = tempfile.mktemp(suffix=".mp3")
    comm = edge_tts.Communicate(text=text, voice=pack["voice"], rate=pack["rate"], pitch=pack["pitch"])
    await comm.save(path)
    return FileResponse(path, media_type="audio/mpeg")

@app.post("/api/video")
async def make_video(image: UploadFile = File(...), audio: UploadFile = File(...)):
    img_path = tempfile.mktemp(suffix=".png")
    aud_path = tempfile.mktemp(suffix=".mp3")
    with open(img_path, "wb") as f:
        f.write(await image.read())
    with open(aud_path, "wb") as f:
        f.write(await audio.read())

    spaces_to_try = ["vinthony/SadTalker", "fffiloni/SadTalker"]
    last_error = ""
    for space_id in spaces_to_try:
        try:
            client = Client(space_id, hf_token=HF_TOKEN)
            result = client.predict(
                source_image=handle_file(img_path),
                driven_audio=handle_file(aud_path),
                preprocess="crop",
                still_mode=False,
                use_enhancer=False,
                api_name="/test"
            )
            video_path = result[0] if isinstance(result, (list, tuple)) else result
            return FileResponse(video_path, media_type="video/mp4")
        except Exception as e:
            last_error = str(e)
            continue

    raise HTTPException(status_code=503, detail=f"वीडियो सर्वर अभी व्यस्त है: {last_error}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
