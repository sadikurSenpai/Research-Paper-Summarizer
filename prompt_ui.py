# gemini-2.0-pro-exp-02-05
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05")


st.header('Research Paper Summarizer')
paper_input = st.selectbox(
    "Choose a Paper to Explore",
    [
        "ImageNet Classification with Deep Convolutional Neural Networks",
        "Playing Atari with Deep Reinforcement Learning",
        "Sequence to Sequence Learning with Neural Networks",
        "Neural Machine Translation by Jointly Learning to Align and Translate",
        "Deep Residual Learning for Image Recognition",
        "YOLO: Real-Time Object Detection",
        "Generative Adversarial Networks",
        "Auto-Encoding Variational Bayes",
        "Distilling the Knowledge in a Neural Network",
        "Understanding Deep Learning Requires Rethinking Generalization",
        "One Model to Learn Them All",
        "Adam: A Method for Stochastic Optimization",
        "Batch Normalization: Accelerating Deep Network Training",
        "Long Short-Term Memory",
        "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
        "XLNet: Generalized Autoregressive Pretraining",
        "T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        "Switch Transformers: Scaling to Trillion Parameter Models",
        "PaLM: Scaling Language Models with Pathways",
        "Segment Anything",
        "SAM: Sharpness-Aware Minimization",
        "EfficientNet: Rethinking Model Scaling for CNNs",
        "MobileNetV2: Inverted Residuals and Linear Bottlenecks",
        "DenseNet: Densely Connected Convolutional Networks",
        "UNet: Convolutional Networks for Biomedical Image Segmentation",
        "BigGAN: Large Scale GAN Training for High Fidelity Natural Image Synthesis",
        "StyleGAN: A Style-Based Generator Architecture for GANs",
        "CLIP: Learning Transferable Visual Models From Natural Language Supervision",
        "DALL·E: Creating Images from Text",
        "Contrastive Learning of Structured World Models",
        "SimCLR: A Simple Framework for Contrastive Learning",
        "MoCo: Momentum Contrast for Unsupervised Visual Representation Learning",
        "BYOL: Bootstrap Your Own Latent",
        "Noisy Student Training",
        "DETR: End-to-End Object Detection with Transformers",
        "Vision Transformers (ViT)",
        "Swin Transformer: Hierarchical Vision Transformer",
        "Reformer: The Efficient Transformer",
        "Perceiver: General Perception with Iterative Attention",
        "AlphaFold: Predicting Protein Structure with AI",
        "MuZero: Mastering Games Without Rules",
        "DreamFusion: Text-to-3D using 2D Diffusion",
        "Diffusion Models: A Comprehensive Survey",
        "Score-Based Generative Modeling",
        "DDPM: Denoising Diffusion Probabilistic Models",
        "NeRF: Representing Scenes as Neural Radiance Fields",
        "LLaMA: Open and Efficient Foundation Models",
        "Phi-2: A Small Language Model With Big Capabilities",
        "Mistral: Efficient and Fast Transformer for LLMs",
        "Gemini: The Multimodal Pathways Model",
    ]
)

style_input  = st.selectbox(
    "Pick a Style of Explanation",
    ["Easy to Understand", "In-Depth Technical", "Code-Focused", "Math-Heavy"]
)

length_input  = st.selectbox(
    "How Detailed Should the Explanation Be?",
    ["Brief (1-2 paragraphs)", "Standard (3-5 paragraphs)", "Comprehensive (full breakdown)"]
)

template = load_prompt('E:\Gen-AI/2_ PROMPTS/template.json')

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)