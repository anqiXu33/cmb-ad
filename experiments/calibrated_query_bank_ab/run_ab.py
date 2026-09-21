"""
A/B compare raw-headshot vs calibrated-centroid character recognition.

Runs face detection ONCE per clip, then matches the same detected faces
against:
  (a) raw headshot embeddings   → column `pred_ids_raw_rerun`
  (b) calibrated centroids      → column `pred_ids_calibrated`

The original `pred_ids` and `bboxes` columns from the input annotation CSV
are preserved untouched. Newly detected bboxes go into `bboxes_rerun`.

This lets us isolate calibration's effect from face-detection variance.

Side experiment — see experiments/calibrated_query_bank_ab/README.md.

Usage:
  python experiments/calibrated_query_bank_ab/run_ab.py \
    --anno_path       <movie>_context_faces_threads_shot_scales.csv \
    --charbank_path   imdb_chars_ext.json \
    --query_bank_path results/preprocess/query_bank/query_bank_0.pkl \
    --video_dir       /path/to/videos/ \
    --movie_imdb_id   tt28132461 \
    --output_path     experiments/calibrated_query_bank_ab/results/Moskau_einfach/pred_ids_ab.csv
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch

from PIL import Image
from decord import VideoReader, cpu
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor, as_completed

_CHAR_RECOG = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "char_recog"
))
sys.path.insert(0, _CHAR_RECOG)
from utils import process_video, group_and_filter_id  # noqa: E402

TOP_K_CHARS = 10


def run_inference(image, index, app):
    return index, app.get(image)


def load_query_bank(path):
    with open(path, "rb") as f:
        qb = pickle.load(f)
    return {
        movie_id: {cid: torch.tensor(emb) for cid, emb in chars.items()}
        for movie_id, chars in qb.items()
    }


def build_raw_features(app, cast_info, allowed_ids):
    """Re-extract headshot embeddings, keeping only IDs in `allowed_ids`."""
    cast_ids, cast_features = [], []
    for char_info in cast_info:
        actor_id = str(char_info.get("id", ""))
        if not actor_id or actor_id not in allowed_ids:
            continue
        img_path = char_info.get("img")
        if not img_path or not os.path.exists(img_path):
            continue
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            faces = app.get(image)
        except Exception as e:
            print(f"  [WARN] headshot fail {actor_id}: {e}")
            continue
        if not faces:
            continue
        feat = torch.tensor(faces[0]["embedding"])
        feat = feat / feat.norm(dim=-1, keepdim=True)
        cast_features.append(feat)
        cast_ids.append(actor_id)
        if len(cast_ids) >= TOP_K_CHARS:
            break
    if not cast_features:
        return [], None
    return cast_ids, torch.stack(cast_features, 0)


def match_faces(face_embeddings, cast_ids, cast_features, score_thresh):
    """Run cosine sim → group_and_filter_id over all frames.

    face_embeddings: list-of-list, frame -> list of (embedding, bbox_int)
    Returns: (bboxes_all_frames, pred_ids_dict)
    """
    pred_id_all_frames, pred_cos_all_frames = [], []
    bboxes_all_frames = []
    for frame_data in face_embeddings:
        if not frame_data:
            bboxes_all_frames.append([])
            pred_id_all_frames.append([])
            pred_cos_all_frames.append([])
            continue
        feats = torch.stack([e for e, _ in frame_data], 0)
        feats_norm = feats / feats.norm(dim=-1, keepdim=True)
        cos_all = feats_norm @ cast_features.transpose(0, 1)
        max_vals, max_idx = cos_all.max(dim=-1)
        bboxes_all_frames.append([b for _, b in frame_data])
        pred_id_all_frames.append([cast_ids[i] for i in max_idx.tolist()])
        pred_cos_all_frames.append(max_vals.tolist())
    pred_ids = group_and_filter_id(pred_id_all_frames, pred_cos_all_frames, score_thresh)
    return bboxes_all_frames, pred_ids


def main(args):
    # ─── load InsightFace ───
    print("Loading InsightFace...")
    app = FaceAnalysis(name="buffalo_l", root=args.cache_dir)
    app.prepare(ctx_id=0, det_thresh=args.det_thresh,
                det_size=(args.det_size, args.det_size))

    # ─── load query bank ───
    qb = load_query_bank(args.query_bank_path)
    if args.movie_imdb_id not in qb:
        print(f"[FATAL] {args.movie_imdb_id} not in {args.query_bank_path}; "
              f"available: {list(qb.keys())}")
        sys.exit(1)
    qb_movie = qb[args.movie_imdb_id]
    calibrated_ids = list(qb_movie.keys())[:TOP_K_CHARS]
    calibrated_features = torch.stack([qb_movie[cid] for cid in calibrated_ids], 0)
    print(f"Query bank: {len(calibrated_ids)} centroids for {args.movie_imdb_id}")

    # ─── load charbank ───
    with open(args.charbank_path) as f:
        charbank = json.load(f)
    cast_info = charbank.get(args.movie_imdb_id)
    if cast_info is None:
        print(f"[FATAL] {args.movie_imdb_id} not in charbank")
        sys.exit(1)

    # build raw features over the SAME actor IDs as the calibrated bank
    raw_ids, raw_features = build_raw_features(app, cast_info, set(calibrated_ids))
    if raw_features is None:
        print("[FATAL] no raw headshot embeddings extracted")
        sys.exit(1)
    print(f"Raw headshots: {len(raw_ids)} embeddings")

    if set(raw_ids) != set(calibrated_ids):
        missing = set(calibrated_ids) - set(raw_ids)
        print(f"  [WARN] {len(missing)} actors in calibrated bank but not raw "
              f"(headshot lost): {missing}")

    # ─── load annotation CSV ───
    anno_df = pd.read_csv(args.anno_path)
    print(f"Annotation: {len(anno_df)} rows")
    if args.movie_imdb_id is not None:
        before = len(anno_df)
        anno_df = anno_df[anno_df["imdbid"].astype(str) == str(args.movie_imdb_id)]
        anno_df = anno_df.reset_index(drop=True)
        print(f"  filtered to {args.movie_imdb_id}: {before} → {len(anno_df)} rows")
    if args.first_n_samples > 0:
        anno_df = anno_df.iloc[:args.first_n_samples]
        print(f"  first_n_samples={args.first_n_samples}")

    # ─── per-clip face detection + dual matching ───
    bboxes_rerun_col, pred_raw_col, pred_calib_col = [], [], []
    last_video, decord_vr = "", None

    for row_idx, row in anno_df.iterrows():
        if row_idx % 50 == 0:
            print(f"  clip {row_idx}/{len(anno_df)}")

        movie_title = row["movie_title"]
        video_path = os.path.join(args.video_dir, movie_title + ".mp4")
        if not os.path.exists(video_path):
            print(f"  [SKIP] no video: {video_path}")
            bboxes_rerun_col.append([]); pred_raw_col.append({}); pred_calib_col.append({})
            continue

        if video_path != last_video:
            try:
                decord_vr = VideoReader(uri=video_path, ctx=cpu(0))
            except Exception as e:
                print(f"  [SKIP] cannot open {video_path}: {e}")
                bboxes_rerun_col.append([]); pred_raw_col.append({}); pred_calib_col.append({})
                continue
            last_video = video_path

        try:
            images = process_video(
                decord_vr,
                start=row["scaled_start"],
                end=row["scaled_end"],
                num_frames=args.num_frames,
            )
        except Exception as e:
            print(f"  [SKIP] decode fail at {row_idx}: {e}")
            bboxes_rerun_col.append([]); pred_raw_col.append({}); pred_calib_col.append({})
            continue

        # parallel face detection
        results = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(run_inference, img, idx, app): idx
                       for idx, img in enumerate(images)}
            for fut in as_completed(futures):
                try:
                    idx, faces = fut.result()
                    results.append((idx, faces))
                except Exception:
                    continue
        results.sort(key=lambda x: x[0])

        # collect (embedding, bbox_int) per frame
        frame_data = []
        for _, faces in results:
            per_frame = []
            for f in faces:
                bbox = f["bbox"].astype(int)
                bbox = np.clip(bbox, a_min=0, a_max=None).tolist()
                bbox_int = [int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])]  # match main.py order
                emb = torch.tensor(f["embedding"])
                per_frame.append((emb, bbox_int))
            frame_data.append(per_frame)

        # dual matching on identical face detections
        bboxes_raw, pred_raw = match_faces(frame_data, raw_ids, raw_features, args.score_thresh)
        _, pred_calib = match_faces(frame_data, calibrated_ids, calibrated_features, args.score_thresh)

        bboxes_rerun_col.append(bboxes_raw)
        pred_raw_col.append(pred_raw)
        pred_calib_col.append(pred_calib)

    anno_df["bboxes_rerun"] = bboxes_rerun_col
    anno_df["pred_ids_raw_rerun"] = pred_raw_col
    anno_df["pred_ids_calibrated"] = pred_calib_col

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    anno_df.to_csv(args.output_path, index=False)
    print(f"\nSaved to {args.output_path}")
    print(f"Columns: {list(anno_df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_path", required=True,
                        help="Original annotation CSV (with `pred_ids` column to preserve)")
    parser.add_argument("--charbank_path", required=True,
                        help="charbank JSON (contains `img` paths to actor headshots)")
    parser.add_argument("--query_bank_path", required=True,
                        help="Calibrated centroid pkl from build_query_bank.py")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--movie_imdb_id", required=True,
                        help="Filter annotation to this IMDB ID, e.g. tt28132461")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--score_thresh", type=float, default=0.2,
                        help="Same default as char_recog/main.py")
    parser.add_argument("--det_thresh", type=float, default=0.4)
    parser.add_argument("--det_size", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--first_n_samples", type=int, default=0,
                        help="Use first N rows only (debug)")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
