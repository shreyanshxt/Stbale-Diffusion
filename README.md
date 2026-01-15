# 🎨 Stable Diffusion Studio

[![React](https://img.shields.io/badge/Frontend-React-61DAFB?logo=react&logoColor=white)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/AI-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tailwind CSS](https://img.shields.io/badge/Styling-Tailwind--CSS-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

A high-performance, full-stack generative AI suite. This application features a **Stable Diffusion v1.4** backend powered by PyTorch and a modern, **Glassmorphism** web interface built with React.



---

## ✨ Features

* **Text-to-Image:** Generate high-fidelity images from natural language descriptions.
* **Image-to-Image:** Transform existing images by combining visual context with text prompts.
* **Advanced Parameter Tuning:** * **CFG Scale:** Adjust how closely the AI follows your prompt.
    * **Inference Steps:** Balance generation quality vs. speed.
    * **Strength Control:** Fine-tune the influence of the source image in `img2img` mode.
    * **Deterministic Seeds:** Replicate specific artistic styles using fixed seeds.
* **Modern UI:** A sleek, dark-themed dashboard with frosted-glass effects, responsive sliders, and real-time generation previews.

---

## 🏗️ Project Structure

```text
.
├── backend/
│   ├── flask-backend.py      # Flask API & Inference Server
│   ├── sd_models.py          # VAE, CLIP, and Diffusion Architecture
│   └── sampler.py            # DDPM/Sampling Logic
└── frontend/
    ├── src/
    │   ├── App.jsx           # Main React UI Component
    │   └── main.jsx          # React Entry Point
    ├── tailwind.config.js    # Styling Configuration
    └── package.json          # Frontend Dependencies
```text
--


