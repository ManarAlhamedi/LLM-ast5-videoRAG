# Functions to load video frames + render them:

import streamlit as st
from PIL import Image
from pathlib import Path
import json

MAPPING_PATH = "data/mappings/frame_text_mapping.json"
KEYFRAMES_DIR = "data/keyframes"

@st.cache_data
def load_mapping():
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return mapping

@st.cache_data
def load_keyframe(frame_name):
    frame_path = Path(KEYFRAMES_DIR) / frame_name
    return Image.open(frame_path)

def find_frame_for_text(matched_text):
    mapping = load_mapping()
    for entry in mapping:
        if entry['text'].strip() == matched_text.strip():
            return entry['frame'], entry['timestamp']
    return None, None
