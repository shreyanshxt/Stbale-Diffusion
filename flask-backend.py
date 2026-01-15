from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import torch
import numpy as np
from transformers import CLIPTokenizer
from pathlib import Path

# Import your model components
# Make sure these imports match your notebook structure
from sd_models import (
    VAE_Encoder, VAE_Decoder, Diffusion, CLIP,
    DDPMSampler, preload_models_from_standard_weights
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables for models
DEVICE = "cpu"
models = None
tokenizer = None

# Configuration
MODEL_PATH = "/Users/ShreyanshSingh/Downloads/sd-v1-4-full-ema.ckpt"
VOCAB_PATH = "/Users/ShreyanshSingh/Downloads/vocab.json"
MERGES_PATH = "/Users/ShreyanshSingh/Downloads/merges.txt"

def initialize_models():
    """Initialize and load all models"""
    global models, tokenizer, DEVICE
    
    # Set device
    ALLOW_CUDA = True
    ALLOW_MPS = False
    
    if torch.cuda.is_available() and ALLOW_CUDA:
        DEVICE = "cuda"
    elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
        DEVICE = "mps"
    
    print(f"Using device: {DEVICE}")
    
    # Load tokenizer
    tokenizer = CLIPTokenizer(VOCAB_PATH, merges_file=MERGES_PATH)
    
    # Load models
    models = preload_models_from_standard_weights(MODEL_PATH, DEVICE)
    
    print("Models loaded successfully!")

def rescale(x, old_range, new_range, clamp=False):
    """Rescale tensor values from one range to another"""
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    """Generate time embedding for diffusion process"""
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def generate_image(
    prompt,
    negative_prompt="",
    input_image=None,
    strength=0.8,
    cfg_scale=7.5,
    steps=50,
    seed=42,
    width=512,
    height=512
):
    """Generate image using Stable Diffusion"""
    
    LATENTS_WIDTH = width // 8
    LATENTS_HEIGHT = height // 8
    
    with torch.no_grad():
        # Initialize generator
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(seed)
        
        # Process text with CLIP
        clip = models["clip"]
        clip.to(DEVICE)
        
        # Conditional prompt
        cond_tokens = tokenizer.batch_encode_plus(
            [prompt], padding="max_length", max_length=77
        ).input_ids
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=DEVICE)
        cond_context = clip(cond_tokens)
        
        # Unconditional prompt
        uncond_tokens = tokenizer.batch_encode_plus(
            [negative_prompt], padding="max_length", max_length=77
        ).input_ids
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=DEVICE)
        uncond_context = clip(uncond_tokens)
        
        # Combine contexts
        context = torch.cat([cond_context, uncond_context])
        clip.to("cpu")
        
        # Initialize sampler
        sampler = DDPMSampler(generator)
        sampler.set_inference_timesteps(steps)
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        
        # Handle input image for img2img
        if input_image is not None:
            encoder = models["encoder"]
            encoder.to(DEVICE)
            
            # Process input image
            input_image_tensor = input_image.resize((width, height))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=DEVICE)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            
            # Encode image
            encoder_noise = torch.randn(latents_shape, generator=generator, device=DEVICE)
            latents = encoder(input_image_tensor, encoder_noise)
            
            # Add noise
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            
            encoder.to("cpu")
        else:
            # Random latents for text2img
            latents = torch.randn(latents_shape, generator=generator, device=DEVICE)
        
        # Diffusion process
        diffusion = models["diffusion"]
        diffusion.to(DEVICE)
        
        for i, timestep in enumerate(sampler.timesteps):
            time_embedding = get_time_embedding(timestep).to(DEVICE)
            model_input = latents.repeat(2, 1, 1, 1)
            
            # Predict noise
            model_output = diffusion(model_input, context, time_embedding)
            
            # Apply CFG
            output_cond, output_uncond = model_output.chunk(2)
            model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            # Denoise step
            latents = sampler.step(timestep, latents, model_output)
            
            # Print progress
            if i % 10 == 0:
                print(f"Step {i}/{steps}")
        
        diffusion.to("cpu")
        
        # Decode latents to image
        decoder = models["decoder"]
        decoder.to(DEVICE)
        images = decoder(latents)
        decoder.to("cpu")
        
        # Post-process
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        
        return images[0]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "device": DEVICE})

@app.route('/generate', methods=['POST'])
def generate():
    """Main endpoint for image generation"""
    try:
        # Get parameters from request
        prompt = request.form.get('prompt', '')
        negative_prompt = request.form.get('negative_prompt', '')
        cfg_scale = float(request.form.get('cfg_scale', 7.5))
        steps = int(request.form.get('steps', 50))
        seed = int(request.form.get('seed', 42))
        width = int(request.form.get('width', 512))
        height = int(request.form.get('height', 512))
        strength = float(request.form.get('strength', 0.8))
        
        # Handle input image if provided
        input_image = None
        if 'input_image' in request.form:
            image_data = request.form.get('input_image')
            # Decode base64 image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            input_image = Image.open(io.BytesIO(image_bytes))
        
        print(f"Generating image with prompt: {prompt}")
        print(f"Parameters: cfg_scale={cfg_scale}, steps={steps}, seed={seed}")
        
        # Generate image
        output_array = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=input_image,
            strength=strength,
            cfg_scale=cfg_scale,
            steps=steps,
            seed=seed,
            width=width,
            height=height
        )
        
        # Convert to PIL Image
        output_image = Image.fromarray(output_array)
        
        # Save to bytes
        img_io = io.BytesIO()
        output_image.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Initializing Stable Diffusion models...")
    initialize_models()
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)