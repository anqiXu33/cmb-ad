"""
Build a Character Best Query Bank via clustering calibration.

Inspired by FocusedAD (Ye et al., 2025): for each character, find the
centroid embedding that maximizes inter-class distance while minimizing
intra-class distance.  This mitigates the domain gap between actor headshots
and their in-movie appearances (different hairstyle, makeup, lighting, etc.).

Pipeline:
  1. Load headshot embeddings for each character (InsightFace).
  2. Scan movie video frames, detect all faces, extract embeddings.
  3. Initial assignment: match each detected face to closest character
     (using headshot embedding + cosine similarity threshold).
  4. For each character, collect all assigned in-movie face embeddings.
  5. Optimize: find centroid per character that maximizes
         f(Q) = Σ_j  D_ext(q_j) / (D_int(q_j) + ε)
     where D_int = mean L2 distance within cluster,
           D_ext = min L2 distance to other cluster centroids.
  6. Save query bank (.pkl) for use in char_recog/main.py.

Usage:
  python char_recog/build_query_bank.py \
    --imdb_chars_path /path/to/imdb_chars_ext.json \
    --actor_images_dir /path/to/actor_images/ \
    --video_dir /path/to/videos/ \
    --anno_dir /path/to/annotations/ \
    --output_path /path/to/query_bank.pkl \
    [--score_thresh 0.2] [--det_thresh 0.4] [--num_frames 16] \
    [--max_frames_per_char 200] [--refinement_rounds 2]
"""

import os
import sys
import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd

from PIL import Image
from decord import VideoReader, cpu
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor, as_completed

TOP_K_CHARS = 10
EPSILON = 1e-6


# ---------------------------------------------------------------
# Face detection helper (same as main.py)
# ---------------------------------------------------------------

def run_inference(image, index, app):
    faces = app.get(image)
    return index, faces


def process_video(decord_vr, num_frames=16, start=None, end=None):
    """Extract uniformly-spaced frames from a video segment."""
    local_fps = float(decord_vr.get_avg_fps())
    start_frame = int(local_fps * start)
    end_frame = min(int(local_fps * end), len(decord_vr) - 1)
    if end_frame <= start_frame:
        return []
    frame_ids = np.linspace(start_frame, end_frame, num_frames,
                            endpoint=False, dtype=int)
    try:
        video_data = decord_vr.get_batch(frame_ids).numpy()
    except Exception:
        video_data = decord_vr.get_batch(frame_ids).asnumpy()
    return [f for f in video_data]


# ---------------------------------------------------------------
# Step 1: Extract headshot embeddings
# ---------------------------------------------------------------

def load_headshot_embeddings(app, chars, actor_images_dir):
    """
    Load headshot images and extract InsightFace embeddings.

    Returns:
        cast_ids:      list of actor ID strings
        cast_features: tensor [N, 512] of normalized embeddings
        failed_ids:    list of actor IDs where no face was detected
    """
    cast_ids = []
    cast_features = []
    failed_ids = []

    for char in chars[:TOP_K_CHARS]:
        actor_id = str(char["id"])
        # Try multiple image naming conventions
        img_path = None
        for candidate in [
            os.path.join(actor_images_dir, f"{actor_id}.jpg"),
            os.path.join(actor_images_dir, f"{char.get('name', '').replace(' ', '_')}.jpg"),
        ]:
            if os.path.exists(candidate):
                img_path = candidate
                break

        # Also try the original path from JSON (cluster path)
        if img_path is None:
            orig = char.get("img", "")
            if orig and os.path.exists(orig):
                img_path = orig

        if img_path is None:
            failed_ids.append(actor_id)
            continue

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            faces = app.get(image)
        except Exception as e:
            print(f"    [WARN] Error reading {img_path}: {e}")
            failed_ids.append(actor_id)
            continue

        if len(faces) == 0:
            failed_ids.append(actor_id)
            continue

        feat = torch.tensor(faces[0]["embedding"])
        feat = feat / feat.norm(dim=-1, keepdim=True)
        cast_ids.append(actor_id)
        cast_features.append(feat)

    if cast_features:
        cast_features = torch.stack(cast_features, 0)
    else:
        cast_features = torch.zeros(0, 512)

    return cast_ids, cast_features, failed_ids


# ---------------------------------------------------------------
# Step 2-3: Collect in-movie face embeddings per character
# ---------------------------------------------------------------

def collect_movie_faces(app, cast_ids, cast_features, video_path,
                        anno_df, score_thresh, num_frames,
                        max_frames_per_char):
    """
    Scan all annotated clips in one movie, detect faces, and assign
    each face to the closest character using headshot embeddings.

    Returns:
        char_embeddings: {actor_id: tensor [M, 512]} of collected
                         in-movie face embeddings per character.
        char_face_crops: {actor_id: list of (embedding, crop_image)} —
                         kept for representative frame selection later.
    """
    char_embeddings = {cid: [] for cid in cast_ids}
    char_face_crops = {cid: [] for cid in cast_ids}  # (embedding, BGR crop)

    last_video = ""
    decord_vr = None
    total_clips = len(anno_df)

    for clip_idx, (_, row) in enumerate(anno_df.iterrows()):
        if clip_idx % 100 == 0:
            print(f"    Scanning clip {clip_idx}/{total_clips}...")

        # Load video frames
        vpath = video_path
        if not os.path.exists(vpath):
            continue

        if vpath != last_video:
            try:
                decord_vr = VideoReader(uri=vpath, ctx=cpu(0))
            except Exception as e:
                print(f"    [WARN] Cannot open {vpath}: {e}")
                continue
            last_video = vpath

        try:
            images = process_video(decord_vr,
                                   start=row["scaled_start"],
                                   end=row["scaled_end"],
                                   num_frames=num_frames)
        except Exception:
            continue

        if not images:
            continue

        # Parallel face detection
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(run_inference, img, idx, app): idx
                for idx, img in enumerate(images)
            }
            for future in as_completed(futures):
                try:
                    idx, faces = future.result()
                    results.append((idx, faces))
                except Exception:
                    continue
        results.sort(key=lambda x: x[0])

        # Match faces to characters
        for frame_idx, faces in results:
            if not faces:
                continue
            feats = [torch.tensor(f["embedding"]) for f in faces]
            feat_stack = torch.stack(feats, 0)
            feat_norm = feat_stack / feat_stack.norm(dim=-1, keepdim=True)
            cos_sim = feat_norm @ cast_features.t()  # [num_faces, num_chars]

            for face_idx in range(len(faces)):
                best_score, best_char_idx = cos_sim[face_idx].max(0)
                if best_score.item() >= score_thresh:
                    cid = cast_ids[best_char_idx.item()]
                    if len(char_embeddings[cid]) < max_frames_per_char:
                        char_embeddings[cid].append(feat_norm[face_idx])
                        # Save face crop for representative frame selection
                        bbox = faces[face_idx]["bbox"].astype(int)
                        img = images[frame_idx]
                        h, w = img.shape[:2]
                        x1, y1, x2, y2 = np.clip(bbox, 0, [w, h, w, h])
                        crop = img[y1:y2, x1:x2]
                        if crop.size > 0:
                            char_face_crops[cid].append(
                                (feat_norm[face_idx], crop)
                            )

    # Stack embeddings
    result = {}
    for cid, embs in char_embeddings.items():
        if embs:
            result[cid] = torch.stack(embs, 0)
            print(f"    Character {cid}: collected {len(embs)} face embeddings")
        else:
            print(f"    Character {cid}: no in-movie faces found, keeping headshot")
    return result, char_face_crops


# ---------------------------------------------------------------
# Step 5: Clustering optimization (FocusedAD formulas 1-3)
# ---------------------------------------------------------------

def compute_intra_distance(embeddings, centroid):
    """D_int(q_j) = mean L2 distance from embeddings to centroid."""
    return (embeddings - centroid.unsqueeze(0)).norm(dim=-1).mean().item()


def compute_inter_distance(centroid, other_centroids):
    """D_ext(q_j) = min L2 distance to any other centroid."""
    if len(other_centroids) == 0:
        return 1.0  # only one character, no inter-class constraint
    dists = (other_centroids - centroid.unsqueeze(0)).norm(dim=-1)
    return dists.min().item()


def optimize_centroids(char_embeddings, cast_ids, cast_features,
                       refinement_rounds=2):
    """
    Find optimal centroid for each character by maximizing:
        f(Q) = Σ_j D_ext(q_j) / (D_int(q_j) + ε)

    Strategy:
      - Initialize each centroid as the mean of the character's
        in-movie face embeddings (minimizes D_int).
      - For characters with no in-movie faces, keep the headshot
        embedding as centroid.
      - Iteratively refine by re-assigning ambiguous faces and
        re-computing centroids.

    Returns:
        centroids: {actor_id: tensor [512]} normalized centroid embedding
    """
    centroids = {}

    # Initialize centroids
    for i, cid in enumerate(cast_ids):
        if cid in char_embeddings and len(char_embeddings[cid]) >= 3:
            # Mean of in-movie faces
            mean_emb = char_embeddings[cid].mean(dim=0)
            centroids[cid] = mean_emb / mean_emb.norm()
        else:
            # Fallback to headshot
            centroids[cid] = cast_features[i]

    # Iterative refinement
    for round_idx in range(refinement_rounds):
        # Re-assign: for each character's embeddings, remove outliers
        # that are closer to another character's centroid
        centroid_ids = list(centroids.keys())
        centroid_stack = torch.stack([centroids[cid] for cid in centroid_ids], 0)

        reassigned = 0
        for cid in centroid_ids:
            if cid not in char_embeddings or len(char_embeddings[cid]) < 3:
                continue

            embs = char_embeddings[cid]
            # Compute distance to all centroids
            dists = (embs.unsqueeze(1) - centroid_stack.unsqueeze(0)).norm(dim=-1)
            # [M, J] — distance from each embedding to each centroid
            cid_idx = centroid_ids.index(cid)
            own_dist = dists[:, cid_idx]
            min_other_dist = torch.cat([dists[:, :cid_idx], dists[:, cid_idx+1:]], dim=1)

            if min_other_dist.shape[1] > 0:
                closest_other = min_other_dist.min(dim=1)[0]
                # Keep only embeddings closer to own centroid than any other
                keep_mask = own_dist <= closest_other
                if keep_mask.sum() >= 3:
                    removed = (~keep_mask).sum().item()
                    reassigned += removed
                    char_embeddings[cid] = embs[keep_mask]

        # Re-compute centroids after filtering
        for cid in centroid_ids:
            if cid in char_embeddings and len(char_embeddings[cid]) >= 3:
                mean_emb = char_embeddings[cid].mean(dim=0)
                centroids[cid] = mean_emb / mean_emb.norm()

        if reassigned > 0:
            print(f"    Round {round_idx + 1}: reassigned {reassigned} faces")
        else:
            print(f"    Round {round_idx + 1}: converged (no reassignments)")
            break

    # Report final objective value
    centroid_ids = list(centroids.keys())
    centroid_stack = torch.stack([centroids[cid] for cid in centroid_ids], 0)
    total_f = 0.0
    for i, cid in enumerate(centroid_ids):
        if cid in char_embeddings and len(char_embeddings[cid]) >= 3:
            d_int = compute_intra_distance(char_embeddings[cid], centroids[cid])
            others = torch.cat([centroid_stack[:i], centroid_stack[i+1:]], 0)
            d_ext = compute_inter_distance(centroids[cid], others)
            score = d_ext / (d_int + EPSILON)
            total_f += score
            print(f"    {cid}: D_int={d_int:.4f}, D_ext={d_ext:.4f}, "
                  f"score={score:.2f}, n_faces={len(char_embeddings[cid])}")
    print(f"    Total f(Q) = {total_f:.2f}")

    return centroids


# ---------------------------------------------------------------
# Step 6: Save representative face crops (closest to centroid)
# ---------------------------------------------------------------

def save_representative_crops(centroids, char_face_crops, output_dir,
                              movie_id, top_k=3):
    """
    For each character, find the top_k face crops whose embeddings are
    closest to the optimized centroid, and save them as JPEG images.

    Returns:
        saved_paths: {actor_id: [path1, path2, ...]}
    """
    saved_paths = {}
    movie_dir = os.path.join(output_dir, movie_id)
    os.makedirs(movie_dir, exist_ok=True)

    for cid, centroid in centroids.items():
        crops_data = char_face_crops.get(cid, [])
        if not crops_data:
            continue

        # Greedy diverse selection: pick crops close to centroid but
        # visually different from each other (avoid near-duplicate frames)
        embs = torch.stack([e for e, _ in crops_data], 0)
        cos_sim = (embs @ centroid.unsqueeze(-1)).squeeze(-1)
        k = min(top_k, len(crops_data))

        # Start with the closest to centroid
        best_idx = cos_sim.argmax().item()
        topk_indices = [best_idx]

        # Greedily pick next crop: must be close to centroid AND
        # far from all already-selected crops
        min_diversity = 0.3  # minimum L2 distance from selected crops
        for _ in range(k - 1):
            best_score = -1
            best_next = -1
            for j in range(len(crops_data)):
                if j in topk_indices:
                    continue
                # Check diversity: L2 distance to all selected
                min_dist = min(
                    (embs[j] - embs[s]).norm().item() for s in topk_indices
                )
                if min_dist < min_diversity:
                    continue
                # Score: cosine similarity to centroid
                if cos_sim[j].item() > best_score:
                    best_score = cos_sim[j].item()
                    best_next = j
            if best_next == -1:
                # Relax diversity constraint if no candidate found
                for j in range(len(crops_data)):
                    if j in topk_indices and cos_sim[j].item() > best_score:
                        continue
                    if j not in topk_indices and cos_sim[j].item() > best_score:
                        best_score = cos_sim[j].item()
                        best_next = j
            if best_next == -1:
                break
            topk_indices.append(best_next)

        paths = []
        for rank, idx in enumerate(topk_indices):
            _, crop = crops_data[idx]
            fname = f"{cid}_rep{rank}.jpg"
            fpath = os.path.join(movie_dir, fname)
            Image.fromarray(crop).save(fpath, quality=95)
            paths.append(fpath)
        saved_paths[cid] = paths
        print(f"    {cid}: saved {len(paths)} representative crops -> {movie_dir}/")

    return saved_paths


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def build_query_bank(args, app):
    """Build query bank for all movies in imdb_chars_ext.json."""
    with open(args.imdb_chars_path, encoding="utf-8") as f:
        imdb_chars = json.load(f)

    # Load checkpoint if exists
    checkpoint = {}
    if os.path.exists(args.output_path):
        try:
            with open(args.output_path, "rb") as f:
                checkpoint = pickle.load(f)
            print(f"Loaded checkpoint: {list(checkpoint.keys())} already done")
        except Exception:
            checkpoint = {}

    query_bank = dict(checkpoint)  # start from checkpoint
    all_rep_crops = {}

    # Filter to specific movies if requested
    movie_ids = list(imdb_chars.keys())
    if args.movie_ids:
        requested = set(args.movie_ids)
        movie_ids = [mid for mid in movie_ids if mid in requested]
        skipped = requested - set(movie_ids)
        if skipped:
            print(f"[WARN] Movie IDs not found in JSON: {skipped}")

    # Skip already-completed movies
    remaining = [mid for mid in movie_ids if mid not in checkpoint]
    if len(remaining) < len(movie_ids):
        print(f"Skipping {len(movie_ids) - len(remaining)} already-completed movies")
    print(f"Processing {len(remaining)} movies (of {len(movie_ids)} total)")

    for movie_id in remaining:
        chars = imdb_chars[movie_id]
        print(f"\n{'='*60}")
        print(f"Processing movie: {movie_id} ({len(chars)} characters)")
        print(f"{'='*60}")

        # Step 1: Load headshot embeddings
        cast_ids, cast_features, failed = load_headshot_embeddings(
            app, chars, args.actor_images_dir
        )
        if failed:
            print(f"  [WARN] No headshot face for: {failed}")
        if len(cast_ids) == 0:
            print(f"  [SKIP] No valid headshot faces for {movie_id}")
            continue
        print(f"  Loaded {len(cast_ids)} headshot embeddings")

        # Find movie title and video path
        movie_title = None
        anno_files = [
            f for f in os.listdir(args.anno_dir)
            if f.endswith("_context_faces_threads_shot_scales.csv")
        ]
        target_anno = None
        for af in anno_files:
            anno_path = os.path.join(args.anno_dir, af)
            try:
                df_tmp = pd.read_csv(anno_path, nrows=1)
                if "imdbid" in df_tmp.columns and str(df_tmp["imdbid"].iloc[0]) == movie_id:
                    target_anno = anno_path
                    movie_title = df_tmp["movie_title"].iloc[0]
                    break
            except Exception:
                continue

        if target_anno is None:
            print(f"  [SKIP] No annotation CSV found for {movie_id}")
            continue

        anno_df = pd.read_csv(target_anno)
        print(f"  Annotation: {os.path.basename(target_anno)} "
              f"({len(anno_df)} clips)")

        # Find video
        video_path = None
        for ext in [".mp4", ".mov", ".avi", ".mkv"]:
            candidate = os.path.join(args.video_dir, movie_title + ext)
            if os.path.exists(candidate):
                video_path = candidate
                break

        if video_path is None:
            print(f"  [SKIP] No video found for {movie_title}")
            continue
        print(f"  Video: {video_path}")

        # Step 2-3: Collect in-movie face embeddings
        char_embeddings, char_face_crops = collect_movie_faces(
            app, cast_ids, cast_features, video_path, anno_df,
            score_thresh=args.score_thresh,
            num_frames=args.num_frames,
            max_frames_per_char=args.max_frames_per_char,
        )

        # Step 5: Optimize centroids
        centroids = optimize_centroids(
            char_embeddings, cast_ids, cast_features,
            refinement_rounds=args.refinement_rounds,
        )

        # Step 6: Save representative face crops (closest to centroid)
        rep_crops = save_representative_crops(
            centroids, char_face_crops,
            output_dir=args.rep_crops_dir,
            movie_id=movie_id,
            top_k=args.num_rep_crops,
        )
        all_rep_crops[movie_id] = rep_crops

        # Store as plain lists for serialization
        query_bank[movie_id] = {
            cid: centroid.tolist() for cid, centroid in centroids.items()
        }

        # Checkpoint: save after each movie
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "wb") as f:
            pickle.dump(query_bank, f)
        print(f"  [CHECKPOINT] Saved {len(query_bank)} movies to {args.output_path}")

    return query_bank, all_rep_crops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Character Best Query Bank via clustering calibration"
    )
    parser.add_argument("--imdb_chars_path", required=True,
                        help="Path to imdb_chars_ext.json")
    parser.add_argument("--actor_images_dir", required=True,
                        help="Directory containing actor headshot images")
    parser.add_argument("--video_dir", required=True,
                        help="Directory containing movie video files")
    parser.add_argument("--anno_dir", required=True,
                        help="Directory containing annotation CSVs")
    parser.add_argument("--output_path", required=True,
                        help="Output path for query bank (.pkl)")
    parser.add_argument("--score_thresh", type=float, default=0.25,
                        help="Cosine similarity threshold for initial assignment "
                             "(slightly higher than main.py default for cleaner clusters)")
    parser.add_argument("--det_thresh", type=float, default=0.4,
                        help="Face detection confidence threshold")
    parser.add_argument("--det_size", type=int, default=640,
                        help="Image size for face detection")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames per clip for face extraction")
    parser.add_argument("--max_frames_per_char", type=int, default=200,
                        help="Max face embeddings to collect per character")
    parser.add_argument("--refinement_rounds", type=int, default=2,
                        help="Iterative refinement rounds for centroid optimization")
    parser.add_argument("--rep_crops_dir", type=str, default="results/preprocess/rep_crops",
                        help="Directory to save representative face crops per character")
    parser.add_argument("--num_rep_crops", type=int, default=3,
                        help="Number of representative crops to save per character")
    parser.add_argument("--movie_ids", nargs="+", default=None,
                        help="Only process these movie IMDB IDs (e.g. tt10949014). "
                             "If omitted, processes all movies in the JSON.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Path to save/load InsightFace model")
    args = parser.parse_args()

    # Initialize InsightFace
    print("Loading InsightFace model...")
    app = FaceAnalysis(name="buffalo_l", root=args.cache_dir)
    app.prepare(ctx_id=0, det_thresh=args.det_thresh,
                det_size=(args.det_size, args.det_size))

    # Build query bank
    query_bank, all_rep_crops = build_query_bank(args, app)

    # Final save (also serves as final checkpoint)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(query_bank, f)
    print(f"\nQuery bank saved to {args.output_path}")
    print(f"Representative crops saved to {args.rep_crops_dir}")
    print(f"Total movies in bank: {len(query_bank)}")
    for mid, chars in query_bank.items():
        n_crops = sum(len(v) for v in all_rep_crops.get(mid, {}).values())
        print(f"  {mid}: {len(chars)} character centroids, {n_crops} representative crops")
