import os
import sys
import pickle
from typing import List, Dict

import cv2
import numpy as np
import torch
import trimesh
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tqdm import tqdm

# NOTE:
# This file wires together the user's SMPL-X rendering + text indexer code
# into a simple HTTP API that the React app can call instead of Meshy.
#
# It assumes that:
# - The SMPL-X utilities are importable as in the original project:
#     from common.utils.smplx import smplx
#     from common.utils.smplx.smplx.utils import Struct
# - The HOW2SIGN dataset + PKL files live under MODEL_PATH.

# Force usage of local patched smplx (contains fixes for missing keys in .npz)
curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, ".."))
local_smplx_path = os.path.join(project_root, "common", "utils", "smplx")

if os.path.exists(local_smplx_path):
    # Insert at 0 to prioritize local version over any global pip install
    if local_smplx_path not in sys.path:
        sys.path.insert(0, local_smplx_path)
    import smplx
    print(f"DEBUG: Loaded smplx from {os.path.dirname(smplx.__file__)}")
else:
    print(f"WARNING: Local smplx paths not found at {local_smplx_path}. Falling back to global.")
    import smplx


# ---------------------------------------------------------
# Constants / paths – ADJUST THESE TO MATCH YOUR ENV
# ---------------------------------------------------------

PREDEFINED_HEIGHT, PREDEFINED_WIDTH = 720, 1280
PRED_FOCALS = [14921.82254791, 14921.82254791]
PRED_PRINCPTS = [620.60418701, 413.40108109]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Project root (repo root) and public dir (for videos)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PUBLIC_DIR = os.path.join(PROJECT_ROOT, "public")
VIDEOS_DIR = os.path.join(PUBLIC_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Paths for your environment – now using folders inside this project:
# smplx.create() will append "smplx" internally when model_type="smplx",
# so we pass the PROJECT_ROOT here, and expect:
#   <project_root>/smplx/SMPLX_NEUTRAL.npz (and related files)
SMPLX_MODEL_PATH = PROJECT_ROOT
PKL_ROOT = os.path.join(PROJECT_ROOT, "how2sign_pkls_cropTrue_shapeTrue")

# CSV used for retrieval – we use the copy already in the repo root
CSV_PATH = os.path.join(PROJECT_ROOT, "how2sign_realigned_train.csv")


# ---------------------------------------------------------
# SMPL-X model init (from user's code, simplified)
# ---------------------------------------------------------

startup_error = None

try:
    smplx_layer = smplx.create(
        SMPLX_MODEL_PATH,
        model_type="smplx",
        gender="NEUTRAL",
        use_pca=False,
        use_face_contour=False,
        ext="npz",
    ).to(DEVICE)
except Exception as e:
    print(f"Error loading SMPL-X model: {e}")
    import traceback

    traceback.print_exc()
    smplx_layer = None
    startup_error = e  # Capture the error for debugging



def get_coord(
    root_pose,
    body_pose,
    lhand_pose,
    rhand_pose,
    jaw_pose,
    shape,
    expr,
    cam_trans,
    mesh: bool = False,
):
    if smplx_layer is None:
        raise RuntimeError("SMPL-X model not loaded.")

    batch_size = root_pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().to(DEVICE).repeat(batch_size, 1)

    output = smplx_layer(
        betas=shape,
        body_pose=body_pose,
        global_orient=root_pose,
        right_hand_pose=rhand_pose,
        left_hand_pose=lhand_pose,
        jaw_pose=jaw_pose,
        leye_pose=zero_pose,
        reye_pose=zero_pose,
        expression=expr,
    )

    mesh_cam = output.vertices

    if mesh:
        render_mesh_cam = mesh_cam + cam_trans[:, None, :]
        return render_mesh_cam


def render_frame(img, mesh, face, cam_param):
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    import pyrender

    # Define material using pyrender directly
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.125,
        roughnessFactor=0.6,
        baseColorFactor=(0.425, 0.72, 0.8, 1),
    )

    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    mesh_node = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh_node, "mesh")

    focal, princpt = cam_param["focal"], cam_param["princpt"]
    camera = pyrender.IntrinsicsCamera(
        fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1]
    )
    scene.add(camera)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0
    )

    # Lighting
    light = pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]), intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    spot_l = pyrender.SpotLight(
        color=np.ones(3),
        intensity=15.0,
        innerConeAngle=np.pi / 3,
        outerConeAngle=np.pi / 2,
    )
    light_pose[:3, 3] = [1, 2, 2]
    scene.add(spot_l, pose=light_pose)
    light_pose[:3, 3] = [-1, 2, 2]
    scene.add(spot_l, pose=light_pose)

    rgb, depth = renderer.render(
        scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
    )
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]

    img = rgb * valid_mask + img * (1 - valid_mask)
    return img


def put_text(image, text: str):
    org = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    img_height, img_width, _ = image.shape
    x, y = org

    lines: List[str] = []
    line = ""
    for word in text.split(" "):
        word_size, _ = cv2.getTextSize(
            (line + " " + word).strip(), font, font_scale, thickness
        )
        word_width, _ = word_size
        if x + word_width > img_width / 3:
            lines.append(line)
            line = word
        else:
            if line:
                line += " "
            line += word
    lines.append(line)

    for i, line in enumerate(lines):
        y_offset = i * text_height * 1.2
        cv2.putText(
            image,
            line,
            (x, int(y + y_offset)),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def render_video(clips: List[Dict], output_path: str, background_path: str):
    if smplx_layer is None:
        print("Cannot render: SMPL-X model not loaded.")
        return

    img_list: List[np.ndarray] = []

    if os.path.exists(background_path):
        background = cv2.imread(background_path)
    else:
        background = np.zeros(
            (PREDEFINED_HEIGHT, PREDEFINED_WIDTH, 3), dtype=np.uint8
        )

    for clip in clips:
        pkl_path = clip["pkl_path"]
        text_annotation = clip["text"]

        print(f"Processing {pkl_path}...")
        with open(pkl_path, "rb") as f:
            results_dict = pickle.load(f)

        all_pose = results_dict["smplx"]
        all_pose = torch.tensor(all_pose).to(DEVICE)

        g, b, l, r, j, s, exp, cam_trans = (
            all_pose[:, :3],
            all_pose[:, 3:66],
            all_pose[:, 66:111],
            all_pose[:, 111:156],
            all_pose[:, 156:159],
            all_pose[:, 159:169],
            all_pose[:, 169:179],
            all_pose[:, 179:182],
        )

        meshes = (
            get_coord(g, b, l, r, j, s, exp, cam_trans[0][None], mesh=True)
            .detach()
            .cpu()
            .numpy()
        )

        faces = smplx_layer.faces
        if torch.is_tensor(faces):
            faces = faces.detach().cpu().numpy()

        for idx in tqdm(results_dict["total_valid_index"]):
            img = render_frame(
                background.copy(),
                meshes[idx],
                faces,
                {"focal": PRED_FOCALS, "princpt": PRED_PRINCPTS},
            ).astype(np.uint8)
            # put_text(img, text_annotation)
            img_list.append(img)

    print(f"Saving video to {output_path}...")
    # Use a simple MPEG-4 codec that OpenCV can write without external H264 libs
    # Switching to avc1 (H.264) for better browser support
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(
        output_path, fourcc, 24, (PREDEFINED_WIDTH, PREDEFINED_HEIGHT)
    )
    for img in img_list:
        out.write(img)
    out.release()
    print("Done.")


# ---------------------------------------------------------
# Text indexer (Updated with user's logic)
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextIndexer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.load_data()

    def load_data(self):
        """Loads the CSV data and builds the TF-IDF index."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at: {self.csv_path}")
        
        print(f"Loading data from {self.csv_path}...")
        # Load only necessary columns to save memory
        self.df = pd.read_csv(self.csv_path, sep='\t', usecols=['SENTENCE_NAME', 'SENTENCE'])
        
        # Drop rows with missing sentences
        self.df.dropna(subset=['SENTENCE'], inplace=True)
        
        print(f"Indexing {len(self.df)} sentences...")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['SENTENCE'])
        print("Indexing complete.")

    def search(self, query, top_k=1):
        """Searches for the most similar sentences to the query."""
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("Index not built. Call load_data() first.")

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            # Explicit float conversion for JSON serialization safety
            score = float(similarities[idx])
            sentence_name = self.df.iloc[idx]['SENTENCE_NAME']
            sentence_text = self.df.iloc[idx]['SENTENCE']
            results.append({
                'score': score,
                'sentence_name': sentence_name,
                'sentence': sentence_text
            })
            
        return results


# Initialize with the project's CSV path
indexer = TextIndexer(CSV_PATH)


def sentence_name_to_pkl_path(sentence_name: str) -> str:
    """
    Map a SENTENCE_NAME from the CSV to a .pkl file path.

    This is a heuristic and might need adjustment depending on your dataset
    layout. Adjust this to match how your PKL files are organized.
    """
    # Example: SENTENCE_NAME: 'how2sign_train_000001' ->
    #   D:\how2sign_pkls_cropTrue_shapeTrue\how2sign_pkls_cropTrue_shapeTrue\how2sign_train_000001.pkl
    candidate = os.path.join(PKL_ROOT, f"{sentence_name}.pkl")
    if os.path.exists(candidate):
        return candidate

    # Fallback: try under a "pkls" subfolder
    candidate2 = os.path.join(PKL_ROOT, "pkls", f"{sentence_name}.pkl")
    return candidate2


# ---------------------------------------------------------
# FastAPI wiring
# ---------------------------------------------------------


class TextToSignRequest(BaseModel):
    text: str


class TextToSignResponse(BaseModel):
    videoUrl: str
    matchedSentence: str
    sentenceName: str


app = FastAPI(title="Text to Sign SMPL-X Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/videos", StaticFiles(directory=PUBLIC_DIR), name="videos")


@app.post("/text-to-sign", response_model=TextToSignResponse)
def text_to_sign(req: TextToSignRequest):
    if smplx_layer is None:
        raise RuntimeError(f"SMPL-X model not loaded on server. Startup error: {startup_error}")

    query = req.text.strip()
    if not query:
        raise ValueError("Empty query.")

    results = indexer.search(query, top_k=1)
    if not results:
        raise RuntimeError("No results from text indexer.")

    best = results[0]
    sentence_name = best["sentence_name"]
    sentence_text = best["sentence"]

    pkl_path = sentence_name_to_pkl_path(sentence_name)
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL file not found for {sentence_name}: {pkl_path}")

    import time
    timestamp = int(time.time())
    out_filename = f"text_to_sign_{sentence_name}_{timestamp}.mp4"
    out_path = os.path.join(PUBLIC_DIR, out_filename)

    background_path = os.path.join(PUBLIC_DIR, "blender.png")
    render_video(
        clips=[{"pkl_path": pkl_path, "text": sentence_text}],
        output_path=out_path,
        background_path=background_path,
    )



    # Serve via the mounted static files path
    video_url = f"/videos/{out_filename}"

    return TextToSignResponse(
        videoUrl=video_url,
        matchedSentence=sentence_text,
        sentenceName=sentence_name,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)


