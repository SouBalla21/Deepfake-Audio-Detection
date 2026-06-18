"""Premium Streamlit interface for audio deepfake analysis."""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from deepfake_audio.audio import AudioValidationError
from deepfake_audio.config import DEFAULT_MODEL_PATH
from deepfake_audio.inference import AudioDetector

st.set_page_config(
    page_title="Auralis | Deepfake Audio Detector",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#f5f7ff; --muted:#9aa5bd; --violet:#8b5cf6; --cyan:#22d3ee; }
.stApp {
  background:
    radial-gradient(circle at 15% 0%, rgba(139,92,246,.20), transparent 31rem),
    radial-gradient(circle at 90% 25%, rgba(34,211,238,.13), transparent 28rem),
    #080b14;
  color:var(--ink); font-family:'DM Sans',sans-serif;
}
[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"], #MainMenu, footer { visibility:hidden; }
.block-container { max-width:1180px; padding-top:2.3rem; padding-bottom:4rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.03em; }
.eyebrow { color:var(--cyan); font-size:.78rem; font-weight:700; letter-spacing:.18em; text-transform:uppercase; }
.hero-title { font:700 clamp(2.8rem,7vw,5.8rem)/.98 'Space Grotesk'; letter-spacing:-.065em; margin:.7rem 0 1rem; max-width:900px; }
.gradient { background:linear-gradient(95deg,#c4b5fd,#67e8f9); -webkit-background-clip:text; color:transparent; }
.sub { color:var(--muted); font-size:1.1rem; line-height:1.7; max-width:720px; margin-bottom:2rem; }
.glass {
  background:linear-gradient(145deg,rgba(25,30,48,.9),rgba(14,18,31,.8));
  border:1px solid rgba(255,255,255,.09); border-radius:24px; padding:1.35rem 1.5rem;
  box-shadow:0 18px 70px rgba(0,0,0,.28); backdrop-filter:blur(16px);
  animation:rise .55s ease-out both;
}
.result-real { border-top:3px solid #2dd4bf; }
.result-fake { border-top:3px solid #fb7185; }
.result-label { font:700 2.25rem 'Space Grotesk'; margin:.25rem 0; }
.metric-name { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.11em; }
.metric-value { font:600 1.4rem 'Space Grotesk'; }
.trust { display:flex; gap:1rem; flex-wrap:wrap; margin:1.3rem 0 2.3rem; }
.pill { background:rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.08); border-radius:999px; padding:.48rem .8rem; color:#cbd5e1; font-size:.82rem; }
[data-testid="stFileUploader"] {
  background:rgba(12,16,29,.75); border:1px dashed rgba(139,92,246,.65);
  border-radius:22px; padding:.75rem; transition:.25s ease;
}
[data-testid="stFileUploader"]:hover { border-color:var(--cyan); transform:translateY(-2px); }
[data-testid="stFileUploaderDropzone"] { background:transparent; }
.stAudio { border-radius:16px; overflow:hidden; }
@keyframes rise { from {opacity:0; transform:translateY(12px)} to {opacity:1; transform:none} }
@media(max-width:700px){ .block-container{padding:1.4rem 1rem 3rem}.hero-title{font-size:3.25rem}.glass{padding:1rem} }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_detector() -> AudioDetector:
    return AudioDetector(DEFAULT_MODEL_PATH)


def probability_chart(genuine: float, deepfake: float) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=[genuine, deepfake],
            y=["Genuine", "Deepfake"],
            orientation="h",
            marker=dict(color=["#2dd4bf", "#a78bfa"], line=dict(width=0)),
            text=[f"{genuine:.1%}", f"{deepfake:.1%}"],
            textposition="inside",
            insidetextanchor="end",
        )
    )
    figure.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe4f3", family="DM Sans"),
        xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(255,255,255,.06)"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return figure


def waveform_chart(data: bytes) -> go.Figure | None:
    try:
        audio, sample_rate = librosa.load(io.BytesIO(data), sr=8_000, mono=True, duration=12)
    except Exception:
        return None
    stride = max(1, len(audio) // 2_000)
    audio = audio[::stride]
    time = np.arange(len(audio)) * stride / sample_rate
    figure = go.Figure(
        go.Scatter(
            x=time,
            y=audio,
            mode="lines",
            line=dict(color="#67e8f9", width=1),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,.08)",
        )
    )
    figure.update_layout(
        height=190,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9aa5bd"),
        xaxis=dict(title="Time (s)", gridcolor="rgba(255,255,255,.05)"),
        yaxis=dict(visible=False),
    )
    return figure


st.markdown('<div class="eyebrow">Audio forensics, made clear</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title">Hear the truth behind <span class="gradient">every voice.</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub">Upload a recording and Auralis examines cepstral, spectral, '
    "harmonic, and temporal traces left by synthetic speech generation.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="trust"><span class="pill">◈ Local analysis</span>'
    '<span class="pill">◎ WAV + MP3</span><span class="pill">⌁ Segment-aware</span>'
    '<span class="pill">◇ Confidence calibrated</span></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.12, 0.88], gap="large")
with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### Analyze a recording")
    uploaded = st.file_uploader(
        "Drop audio here",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        help="For best results, use at least one second of clear speech.",
        label_visibility="collapsed",
    )
    if uploaded:
        data = uploaded.getvalue()
        st.audio(data, format=uploaded.type)
        wave = waveform_chart(data)
        if wave:
            st.plotly_chart(wave, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption("Maximum analyzed duration: 12 seconds. Your file stays on this machine.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    if not uploaded:
        st.markdown(
            '<div class="glass"><div class="eyebrow">Ready when you are</div>'
            "<h3>One file. A deeper signal check.</h3>"
            '<p class="sub" style="font-size:.96rem;margin:0">Your result will include a '
            "verdict, confidence score, and class probability breakdown.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        try:
            with st.spinner("Inspecting acoustic fingerprints..."):
                result = get_detector().predict(data)
            css_class = "result-fake" if result.label == "Deepfake" else "result-real"
            st.markdown(f'<div class="glass {css_class}">', unsafe_allow_html=True)
            st.markdown('<div class="metric-name">Detection result</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-label">{result.label}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-name">Model confidence</div><div class="metric-value">'
                f"{result.confidence:.1%}</div>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                probability_chart(
                    result.genuine_probability, result.deepfake_probability
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                "This is a forensic screening result, not proof of authorship. "
                "Compression, noise, and unseen generators can affect confidence."
            )
            st.markdown("</div>", unsafe_allow_html=True)
        except FileNotFoundError as exc:
            st.error(str(exc))
        except AudioValidationError as exc:
            st.warning(str(exc))
        except Exception:
            st.error("Analysis failed for this file. Try a clearer WAV or MP3 recording.")

st.markdown("---")
st.caption("Auralis Deepfake Audio Detector · FoR-trained acoustic ensemble · v1.0")

