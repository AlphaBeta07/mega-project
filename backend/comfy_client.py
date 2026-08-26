import urllib.request
import urllib.parse
import json
import uuid
import asyncio
import aiohttp
import os
import random

COMFY_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
COMFY_WS_URL = COMFY_URL.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

def get_default_workflow(prompt: str):
    """
    Returns a basic SD 1.5 workflow mapping for ComfyUI.
    """
    return {
        "3": {
            "inputs": {
                "seed": random.randint(1, 100000000),
                "steps": 20,
                "cfg": 7,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "4": {
            "inputs": {
                "ckpt_name": "v1-5-pruned-emaonly.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "text": f"{prompt}, high quality, beautiful, professional illustration, flat vector style, material design, clean solid background, vivid colors",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": "text, watermark, ugly, deformed, blurry, photo, realistic, 3d, complex, messy",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode"
        },
        "9": {
            "inputs": {
                "filename_prefix": "StudySnap",
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        }
    }

async def check_comfyui_connection() -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{COMFY_URL}/system_stats", timeout=3) as resp:
                return resp.status == 200
    except:
        return False

async def generate_image(prompt: str) -> str:
    """Queue image generation and wait for it to complete. Returns filename."""
    is_connected = await check_comfyui_connection()
    if not is_connected:
        raise Exception("ComfyUI is not running or not reachable.")

    client_id = str(uuid.uuid4())
    workflow = get_default_workflow(prompt)
    
    # 1. Queue prompt
    data = json.dumps({"prompt": workflow, "client_id": client_id}).encode('utf-8')
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data)
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read())
            prompt_id = res_data.get("prompt_id")
    except Exception as e:
        raise Exception(f"Failed to queue prompt: {e}")

    # 2. Wait for completion (Polling method)
    max_retries = 60 # 1 minute max for generation
    while max_retries > 0:
        await asyncio.sleep(1)
        req = urllib.request.Request(f"{COMFY_URL}/history/{prompt_id}")
        try:
            with urllib.request.urlopen(req) as response:
                history = json.loads(response.read())
                if prompt_id in history:
                    # Parse output
                    outputs = history[prompt_id].get("outputs", {})
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            image_info = node_output["images"][0]
                            return image_info["filename"]
        except:
            pass
        max_retries -= 1
        
    raise Exception("Image generation timed out.")

async def get_image_bytes(filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
    """Download the generated image bytes from ComfyUI."""
    params = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    })
    url = f"{COMFY_URL}/view?{params}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                raise Exception(f"Failed to fetch image {filename}")
