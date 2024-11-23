import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Escolha do modelo especializado em anime, como Waifu Diffusion ou Anything V4
model_id = "hakurei/waifu-diffusion"  # Ou substitua pelo modelo de anime escolhido
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carregar o modelo
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

def generate_image(prompt, negative_prompt="low quality, blurry, bad anatomy, bad lighting, cropped, text, watermark"):
    """
    Gera uma imagem de estilo anime com base no prompt fornecido e evita elementos descritos no prompt negativo.
    
    Parâmetros:
        prompt (str): A descrição da imagem desejada.
        negative_prompt (str): A descrição dos elementos indesejados (opcional).
    
    Retorna:
        image (PIL.Image): A imagem gerada.
    """
    # Gera a imagem
    with torch.no_grad():
        with torch.autocast("cuda"):
            result = pipe(prompt, negative_prompt=negative_prompt, guidance_scale=7.5)  # Ajuste do guidance_scale para mais estilo
            image = result.images[0]
    return image

# Interface com Gradio para receber prompt e prompt negativo
gr.Interface(
    fn=generate_image,
    inputs=[gr.Textbox(label="Prompt"), gr.Textbox(label="Negative Prompt", placeholder="e.g., low quality, blurry")],
    outputs="image"
).launch()
