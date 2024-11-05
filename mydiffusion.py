import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Carregar o modelo Stable Diffusion da Hugging Face
model_id = "CompVis/stable-diffusion-v1-4"  # Modelo estável
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pipeline para geração de imagem
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

def generate_image(prompt):
    # Gera a imagem
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]
    return image

# Interface usando Gradio para um acesso fácil ao servidor
gr.Interface(fn=generate_image, inputs="text", outputs="image").launch()
