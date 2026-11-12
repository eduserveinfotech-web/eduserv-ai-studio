import os
import json
import asyncio
import tempfile
import gradio as gr
import edge_tts
from huggingface_hub import InferenceClient
from gradio_client import Client, handle_file
from google import genai

HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# 6 अलग-अलग इमोशनल वॉयस पैक्स
CHARACTER_PACKS = {
    "Gussail Chacha":  {"voice": "hi-IN-MadhurNeural", "rate": "+10%", "pitch": "-18Hz"},
    "Chulbuli Dadi":   {"voice": "hi-IN-SwaraNeural",  "rate": "+16%", "pitch": "+24Hz"},
    "Filmi Villain":   {"voice": "hi-IN-MadhurNeural", "rate": "-12%", "pitch": "-30Hz"},
    "Cartoon Bacha":   {"voice": "hi-IN-SwaraNeural",  "rate": "+22%", "pitch": "+42Hz"},
    "Darawna Bhoot":   {"voice": "hi-IN-MadhurNeural", "rate": "-22%", "pitch": "-38Hz"},
    "Shararti Dost":   {"voice": "hi-IN-MadhurNeural", "rate": "+18%", "pitch": "+16Hz"},
}

def plan_reel(idea: str):
    """Gemini AI से वायरल हिंदी डायलॉग और 3D कैरेक्टर प्रॉम्प्ट तैयार करना"""
    if not GEMINI_KEY:
        return idea, f"{idea}, funny 3D talking character with eyes and mouth", "Comedy"
    
    client = genai.Client(api_key=GEMINI_KEY)
    prompt = f"""
    You are a viral Hindi Instagram Reels & YouTube Shorts creator.
    User Idea: "{idea}"
    Create a hilarious, expressive short script. Return ONLY valid JSON with these exact 3 keys:
    1. "dialogue": Punchy, dramatic Hindi dialogue (20 to 35 words) using natural pauses like '...' and exclamation '!' so text-to-speech sounds super expressive.
    2. "image_prompt": Detailed English prompt for a 3D Pixar-style character (vegetable, snake, animal, or object) with big expressive human-like eyes and clear mouth facing the camera.
    3. "emotion": Choose one from ["Comedy", "Angry", "Dramatic", "Excited"].
    """
    try:
        res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        clean = res.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        return data["dialogue"], data["image_prompt"], data["emotion"]
    except Exception:
        return idea, f"{idea}, cute 3D Pixar style character with big eyes and mouth", "Comedy"

def make_image(prompt: str):
    """FLUX.1-schnell मॉडल से अल्ट्रा-क्लियर 3D कैरेक्टर बनाना"""
    client = InferenceClient(token=HF_TOKEN)
    enhanced = (
        f"{prompt}, 3D Pixar and Dreamworks animation style, big expressive eyes, "
        "clearly visible expressive mouth, front-facing portrait, centered, vibrant cinematic lighting, 8k"
    )
    img = client.text_to_image(enhanced, model="black-forest-labs/FLUX.1-schnell")
    path = tempfile.mktemp(suffix=".png")
    img.save(path)
    return path

def make_voice(text: str, emotion: str, voice_pack: str):
    """चुने गए किरदार के अंदाज में इमोशनल हिंदी आवाज बनाना"""
    pack = CHARACTER_PACKS.get(voice_pack, CHARACTER_PACKS["Gussail Chacha"])
    path = tempfile.mktemp(suffix=".mp3")
    
    async def _run():
        comm = edge_tts.Communicate(
            text=text,
            voice=pack["voice"],
            rate=pack["rate"],
            pitch=pack["pitch"]
        )
        await comm.save(path)
        
    asyncio.run(_run())
    return path

def make_video(image, audio):
    """इमेज और ऑडियो को मिलाकर बोलता हुआ लिप-सिंक वीडियो बनाना (Smart Multi-Server Fallback)"""
    custom_space = os.getenv("VIDEO_SPACE", "").strip()
    spaces_to_try = [s for s in [custom_space, "vinthony/SadTalker", "fffiloni/SadTalker"] if s]
    
    last_error = ""
    for space_id in spaces_to_try:
        try:
            client = Client(space_id, hf_token=HF_TOKEN)
            result = client.predict(
                source_image=handle_file(image),
                driven_audio=handle_file(audio),
                preprocess="crop",
                still_mode=False,
                use_enhancer=False,
                api_name="/test"
            )
            return result[0] if isinstance(result, (list, tuple)) else result
        except Exception as e:
            last_error = str(e)
            continue
            
    raise gr.Error(f"सभी वीडियो सर्वर अभी व्यस्त हैं, कृपया 1 मिनट बाद पुनः प्रयास करें। ({last_error})")

with gr.Blocks(title="Kirdar AI Studio Backend") as demo:
    gr.Markdown("# 🎬 किरदार AI स्टूडियो - Cloud API")
    
    idea_in = gr.Textbox(label="Idea")
    diag_out = gr.Textbox(label="Dialogue")
    img_p_out = gr.Textbox(label="Image Prompt")
    emo_out = gr.Textbox(label="Emotion")
    gr.Button("Plan").click(plan_reel, [idea_in], [diag_out, img_p_out, emo_out], api_name="plan")

    img_out = gr.Image(type="filepath", label="Image")
    gr.Button("Image").click(make_image, [img_p_out], [img_out], api_name="image")

    pack_in = gr.Textbox(value="Gussail Chacha", label="Pack")
    aud_out = gr.Audio(type="filepath", label="Audio")
    gr.Button("Voice").click(make_voice, [diag_out, emo_out, pack_in], [aud_out], api_name="voice")

    vid_out = gr.Video(label="Video")
    gr.Button("Video").click(make_video, [img_out, aud_out], [vid_out], api_name="video")

demo.queue(max_size=25).launch()
