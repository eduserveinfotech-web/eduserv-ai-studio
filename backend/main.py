import os
import re
import glob
import json
import math
import time
import tempfile
import subprocess
import urllib.request
import urllib.parse
import edge_tts
import imageio_ffmpeg
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from huggingface_hub import InferenceClient
from google import genai

app = FastAPI(title="Eduserv AI Studio - Pro Dynamic Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server Error: {str(exc)}"},
        headers={"Access-Control-Allow-Origin": "*"},
    )

HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

CHARACTER_PACKS = {
    "Gussail Chacha":  {"voice": "hi-IN-MadhurNeural", "rate": "+12%", "pitch": "-16Hz", "style": "angry grumpy old Indian uncle"},
    "Chulbuli Dadi":   {"voice": "hi-IN-SwaraNeural",  "rate": "+16%", "pitch": "+24Hz", "style": "funny energetic Indian grandmother"},
    "Filmi Villain":   {"voice": "hi-IN-MadhurNeural", "rate": "-10%", "pitch": "-28Hz", "style": "dramatic Bollywood villain"},
    "Cartoon Bacha":   {"voice": "hi-IN-SwaraNeural",  "rate": "+22%", "pitch": "+40Hz", "style": "cute mischievous cartoon kid"},
    "Darawna Bhoot":   {"voice": "hi-IN-MadhurNeural", "rate": "-20%", "pitch": "-36Hz", "style": "spooky yet funny ghost"},
    "Shararti Dost":   {"voice": "hi-IN-MadhurNeural", "rate": "+18%", "pitch": "+15Hz", "style": "energetic funny best friend"},
}

def cleanup_files(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

def clean_old_temp_files():
    """पुराने अस्थायी मीडिया को साफ रखता है ताकि सर्वर की RAM/Disk कभी न भरे"""
    now = time.time()
    tmp_dir = tempfile.gettempdir()
    for pattern in ("*.png", "*.mp3", "*.mp4"):
        for f in glob.glob(os.path.join(tmp_dir, pattern)):
            try:
                if now - os.path.getmtime(f) > 300:
                    os.remove(f)
            except Exception:
                pass

@app.get("/")
def home():
    return {
        "status": "online",
        "engine": "Eduserv Dynamic Reel Engine v3.0",
        "hf_token_loaded": bool(HF_TOKEN),
        "gemini_loaded": bool(GEMINI_KEY),
    }

@app.post("/api/plan")
async def plan_reel(idea: str = Form(...)):
    clean_old_temp_files()
    if not GEMINI_KEY:
        return {
            "dialogue": f"अरे सुनो भाई! {idea}... अब तो कमाल ही हो गया!",
            "image_prompt": f"{idea}, expressive 3D Pixar talking character, big eyes, open mouth",
            "emotion": "Comedy",
        }

    client = genai.Client(api_key=GEMINI_KEY)
    prompt = f"""
    You are India's #1 viral Hindi Reels & Shorts scriptwriter.
    User Idea: "{idea}"
    Create a hilarious, punchy, dramatic Hindi reel script.
    Return ONLY raw valid JSON (no markdown) with exact 3 keys:
    1. "dialogue": High-energy, natural Hinglish/Hindi dialogue (22-38 words) with dramatic pauses '...', exclamations '!', and a punchy ending.
    2. "image_prompt": Detailed English prompt for a 3D Pixar/Disney style expressive character matching the idea (e.g., anthropomorphic vegetable, animal, object, or funny person) with huge expressive eyes, vibrant colors, dramatic lighting, facing directly at the camera.
    3. "emotion": Strictly one of ["Comedy", "Angry", "Dramatic", "Excited"].
    """
    try:
        res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw = re.sub(r"```json|```", "", res.text).strip()
        data = json.loads(raw)
        return JSONResponse(content={
            "dialogue": data.get("dialogue", idea),
            "image_prompt": data.get("image_prompt", f"{idea}, 3D Pixar character"),
            "emotion": data.get("emotion", "Comedy"),
        })
    except Exception:
        return {
            "dialogue": idea,
            "image_prompt": f"{idea}, cute 3D Pixar character, big expressive eyes",
            "emotion": "Comedy",
        }

@app.post("/api/image")
async def make_image(prompt: str = Form(...), background_tasks: BackgroundTasks = None):
    enhanced = (
        f"{prompt}, 3D Pixar animation masterpiece, huge expressive cartoon eyes, "
        "expressive talking face, vibrant studio lighting, centered portrait, 9:16 vertical composition, 8k resolution"
    )
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)

    # इंजन 1: Hugging Face FLUX.1-schnell
    if HF_TOKEN:
        for model_id in ["black-forest-labs/FLUX.1-schnell", "stabilityai/stable-diffusion-xl-base-1.0"]:
            try:
                client = InferenceClient(token=HF_TOKEN)
                img = client.text_to_image(enhanced, model=model_id)
                img.save(path)
                if background_tasks:
                    background_tasks.add_task(cleanup_files, path)
                return FileResponse(path, media_type="image/png")
            except Exception:
                continue

    # इंजन 2: Pollinations Flux (Fast No-Limit Fallback)
    try:
        encoded = urllib.parse.quote(enhanced)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=720&height=1280&model=flux&nologo=true&seed={int(time.time()) % 10000}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            with open(path, "wb") as f:
                f.write(resp.read())
        if background_tasks:
            background_tasks.add_task(cleanup_files, path)
        return FileResponse(path, media_type="image/png")
    except Exception as e:
        cleanup_files(path)
        raise HTTPException(status_code=502, detail=f"Image generation failed: {str(e)}")

@app.post("/api/voice")
async def make_voice(
    text: str = Form(...),
    voice_pack: str = Form("Gussail Chacha"),
    background_tasks: BackgroundTasks = None,
):
    pack = CHARACTER_PACKS.get(voice_pack, CHARACTER_PACKS["Gussail Chacha"])
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        comm = edge_tts.Communicate(
            text=text,
            voice=pack["voice"],
            rate=pack["rate"],
            pitch=pack["pitch"],
        )
        await comm.save(path)
        if background_tasks:
            background_tasks.add_task(cleanup_files, path)
        return FileResponse(path, media_type="audio/mpeg")
    except Exception as e:
        cleanup_files(path)
        raise HTTPException(status_code=500, detail=f"Voice generation failed: {str(e)}")

@app.post("/api/video")
async def make_video(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    fd1, img_path = tempfile.mkstemp(suffix=".png")
    fd2, aud_path = tempfile.mkstemp(suffix=".mp3")
    fd3, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd1)
    os.close(fd2)
    os.close(fd3)

    with open(img_path, "wb") as f:
        f.write(await image.read())
    with open(aud_path, "wb") as f:
        f.write(await audio.read())

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # डायनेमिक मोशन फ़िल्टर: कैरेक्टर में जीवंत कैमरा जूम + बोलने जैसा हल्का वर्टिकल बाउंस मोशन
    # हल्का CPU फ़िल्टर जो 512MB RAM में भी 3-4 सेकंड में HD MP4 रील बनाता है
    dynamic_filter = (
        "scale=800:1422:force_original_aspect_ratio=increase,"
        "crop=800:1422,"
        "crop=720:1280:'40+25*sin(t*1.2)':'71+18*abs(sin(t*6.5))',"
        "format=yuv420p"
    )

    try:
        cmd = [
            ffmpeg_exe, "-y",
            "-loop", "1", "-framerate", "24", "-i", img_path,
            "-i", aud_path,
            "-vf", dynamic_filter,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", out_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        if background_tasks:
            background_tasks.add_task(cleanup_files, img_path, aud_path, out_path)
        return FileResponse(out_path, media_type="video/mp4")
    except Exception:
        # अल्ट्रा-सेफ बैकअप फ़िल्टर (यदि किसी इमेज का अनुपात अलग हो)
        try:
            safe_filter = "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
            cmd_safe = [
                ffmpeg_exe, "-y",
                "-loop", "1", "-framerate", "20", "-i", img_path,
                "-i", aud_path,
                "-vf", safe_filter,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "25",
                "-c:a", "aac", "-b:a", "128k",
                "-shortest", out_path,
            ]
            subprocess.run(cmd_safe, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25)
            if background_tasks:
                background_tasks.add_task(cleanup_files, img_path, aud_path, out_path)
            return FileResponse(out_path, media_type="video/mp4")
        except Exception as e2:
            cleanup_files(img_path, aud_path, out_path)
            raise HTTPException(status_code=500, detail=f"Video render error: {str(e2)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
