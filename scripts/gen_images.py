"""Генерация иллюстраций лендинга моделью FLUX.1-Krea (Hugging Face Spaces). Нужен HF-токен."""
import os, shutil, sys, pathlib
from gradio_client import Client
from PIL import Image

OUT = pathlib.Path(__file__).resolve().parents[1] / "site" / "img"
STYLE = ("3D render, pure black background, translucent olive-lime green glass material with subtle internal glow, "
         "soft cinematic green rim light, clean smooth surfaces, no particles, no dots, no sparkles, glossy dark reflective floor, "
         "shallow depth of field, premium minimalist tech illustration, no text, no logos")
SCENES = {
    "hero": "four irregular rounded translucent glass plates of different sizes floating side by side like puzzle pieces, each holding a cluster of glossy glass spheres connected by thin glowing threads, threads also link the clusters to each other, low camera angle",
    "network": "three stacked transparent glass layers, each with a different web of glowing nodes and edges, merging into one network on top",
    "methods": "several glass podiums of different heights with glass spheres and cubes on them, like a comparison of models, measuring rings around",
    "quality": "glass cubes with glowing scale marks and a balance-like glass structure, precise measurement concept",
    "dynamics": "flowing ribbons of translucent green glass streaming between clusters of glass spheres, sense of motion over time",
    "cases": "a magnifying glass lens made of green glass hovering over small glass buildings and a river",
    "type0": "small village of low glass houses among a field of glass wheat, a narrow country road, wide open space",
    "type1": "glass factory with chimneys and gears next to apartment blocks",
    "type2": "small town square with a two-storey glass town hall with a clock tower, a few low houses and a parcel locker",
    "type3": "lonely glass highway with a fuel station and fields on both sides, receding into distance",
    "type4": "dense cluster of tall smooth glass skyscrapers with solid glowing edges, no windows lights, wide empty plaza in front",
    "type5": "glass ice floes, an oil derrick and a cargo ship in a cold northern landscape",
}

def main(names):
    os.environ.setdefault("HF_TOKEN", open(os.path.expanduser("~/.cache/huggingface/token")).read().strip())
    c = Client("black-forest-labs/FLUX.1-Krea-dev")
    OUT.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        w, h = (1536, 768) if name in ("hero",) else (1024, 640)
        prompt = f"{SCENES[name]}. {STYLE}"
        try:
            r = c.predict(prompt=prompt, seed=11 + i, randomize_seed=False, width=w, height=h, guidance_scale=4.5, num_inference_steps=28, api_name="/infer")
        except Exception as e:
            print(name, "ошибка:", str(e)[:200]); continue
        Image.open(r[0]).convert("RGB").save(OUT / f"{name}.webp", "WEBP", quality=82)
        print(name, "ok")

if __name__ == "__main__":
    main(sys.argv[1:] or list(SCENES))
