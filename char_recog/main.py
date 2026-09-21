import os
import ast
import cv2
import sys
import json 
import torch
import pickle
import argparse
import pandas as pd
import numpy as np

from PIL import Image
from decord import VideoReader, cpu

from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import read_tarfile, process_video, group_and_filter_id

def run_inference(image, index, app):
    faces = app.get(image)
    return index, faces

def load_query_bank(path):
    """Load pre-computed query bank (from build_query_bank.py)."""
    with open(path, "rb") as f:
        qb = pickle.load(f)
    # Convert lists back to tensors
    result = {}
    for movie_id, chars in qb.items():
        result[movie_id] = {
            cid: torch.tensor(emb) for cid, emb in chars.items()
        }
    return result

def main(args, app, charbank_dict, anno_df, anno_df_with_face, query_bank=None):

    # for more efficient cmd video reading
    last_video = ""  
    # cache profile features 
    checked_cast_dicts = {}  

    # iterate over all clips
    for anno_idx, anno_row in anno_df.iterrows():
        # ------ load cast information ------ 
        if args.dataset == "tvad":
            imdbid = anno_row['imdbid_pe'] # use per-episode imdbid for tvad
            imdb_dump = charbank_dict.get(imdbid, None)
            if imdb_dump is None:
                continue
            cast_info = imdb_dump
        elif args.dataset == "cmdad":
            imdbid = anno_row['imdbid']
            imdb_dump = charbank_dict.get(imdbid, None)
            if imdb_dump is None:
                continue
            cast_info = imdb_dump
        elif args.dataset == "madeval":
            imdbid = anno_row['imdb']
            movie_name = anno_row['movie']
            charbank_dict_per_movie = charbank_dict[movie_name]
            cast_info = [{"id": nmid} for nmid in charbank_dict_per_movie["charbank_nmids"].iloc[0]]
        elif args.dataset == "swissAD":
            imdbid = anno_row['imdbid']
            movie_name = anno_row['movie_title']
            imdb_dump = charbank_dict.get(imdbid, None)
            if imdb_dump is None:
                continue
            cast_info = imdb_dump
        else:
            print("Check dataset")
            sys.exit()

        # ------ extract protrait (profile) face features -------
        if imdbid in checked_cast_dicts.keys():
            cast_ids, cast_features = checked_cast_dicts[imdbid]
        elif query_bank is not None and imdbid in query_bank:
            # Use pre-computed query bank centroids (from build_query_bank.py)
            qb_movie = query_bank[imdbid]
            cast_ids = list(qb_movie.keys())[:TOP_K_CHARS]
            cast_features = [qb_movie[cid] for cid in cast_ids]
            print(f"  Using query bank for {imdbid}: {len(cast_ids)} centroids")
        else:
            cast_ids = []
            cast_features = []
            for char_info in cast_info:
                if len(cast_ids) >= TOP_K_CHARS:  # stop when there are topk (10) characters
                    continue
                if char_info['img'] is not None:
                    try:
                        image = np.array(Image.open(char_info['img']).convert("RGB"))
                        faces = app.get(image)
                    except:
                        print("Error in face detection, continue")
                        continue
                    # extract single face feature for a profile image (profile face feature)
                    feats = [torch.tensor(i['embedding']) for i in faces]
                    if len(feats) != 0:
                        feat_frame = feats[0]
                        feat_frame_norm = feat_frame / feat_frame.norm(dim=-1, keepdim=True)
                        cast_features.append(feat_frame_norm)
                        cast_ids.append(char_info['id'])
                    else:
                        continue
        if len(cast_ids) == 0: # no profile face available for the cast list
            anno_df_with_face = anno_df_with_face.append(anno_row)
            continue
        else:
            cast_features = torch.stack(cast_features, 0)
        
        # ------ load images (16 frames) ------
        if args.dataset == "tvad":
            images = []
            seg_name = anno_row["tvad_name"]
            if "friends" in seg_name:
                seg_path = os.path.join(args.video_dir, "friends_frames", seg_name + ".tar")
            else:
                seg_path = os.path.join(args.video_dir, "bbt_frames", seg_name + ".tar")
            for tvad_idx in ast.literal_eval(anno_row["tvad_index"]):
                image_perframe = read_tarfile(seg_path, tvad_idx)
                images.append(np.array(image_perframe))
        elif args.dataset == "cmdad":
            cmd_filename = anno_row['cmd_filename']
            video_path = os.path.join(args.video_dir, cmd_filename + '.mkv')
            if video_path != last_video:
                decord_vr = VideoReader(uri=video_path, ctx=cpu(0)) 
                last_video = video_path
            images = process_video(decord_vr, start = anno_row['scaled_start'], end = anno_row['scaled_end'])
        elif args.dataset == "madeval":
            lsmdc_filename = anno_row['lsmdc_filename']
            start = 0
            end = anno_row['end'] - anno_row['start']
            video_path = os.path.join(args.video_dir, movie_name, lsmdc_filename + '.avi')
            decord_vr = VideoReader(uri=video_path, ctx=cpu(0)) 
            images = process_video(decord_vr, start = start, end = end)
        elif args.dataset == "swissAD":
            filename = anno_row['movie_title']
            video_path = os.path.join(args.video_dir, filename + ".mp4")
            print("video path ", video_path)
            if video_path != last_video:
                decord_vr = VideoReader(uri=video_path, ctx=cpu(0))
                last_video = video_path
            images = process_video(decord_vr, start = anno_row['scaled_start'], end = anno_row['scaled_end'], num_frames=args.num_frames)
        else:
            print("Check dataset")
            sys.exit()
        
        # ------ extract in-frame face features ------
        # parallel face detection to get "faces" for all frames
        results_all_frames = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(run_inference, image, index, app): index for index, image in enumerate(images)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    _, faces = future.result()
                    results_all_frames.append((index, faces))
                except Exception as exc:
                    print(f"Image processing generated an exception: {exc}")
                    continue
        results_all_frames.sort(key=lambda x: x[0])
        faces_all_frames = [faces for _, faces in results_all_frames]

        bboxes_all_frames = []    # bounding boxes for all frames
        pred_id_all_frames = []   # predicted actors id for all frames
        pred_cos_all_frames = []  # predicted cosine similarity values for all frames
        for image_idx, image in enumerate(images):
            try:
                faces = faces_all_frames[image_idx] 
            except:
                continue
            bboxes = [i['bbox'].tolist() for i in faces]
            bboxes = [[int(y1), int(x1), int(y2), int(x2)] for [y1,x1,y2,x2] in bboxes]
            bboxes = np.clip(np.array(bboxes), a_min=0, a_max=None).tolist()
            if len(bboxes) == 0:   # no face detected
                bboxes_all_frames.append([])
                pred_id_all_frames.append([])
                pred_cos_all_frames.append([])
                continue
            else:
                bboxes_all_frames.append(bboxes)
                feats = [torch.tensor(i['embedding']) for i in faces]
                feat_frame = torch.stack(feats, dim=0)
                feat_frame_norm = feat_frame / feat_frame.norm(dim=-1, keepdim=True)
                cos_all = feat_frame_norm @ cast_features.transpose(0, 1)
                pred_id_all_frames.append([cast_ids[idx] for idx in torch.max(cos_all, -1)[1].tolist()])
                pred_cos_all_frames.append([value for value in torch.max(cos_all, -1)[0].tolist()])
        
        # match, filter and format 
        index_filtered_dict = group_and_filter_id(pred_id_all_frames, pred_cos_all_frames, args.score_thresh)
        
        anno_row["bboxes"] = bboxes_all_frames
        anno_row["pred_ids"] = index_filtered_dict
        anno_df_with_face = pd.concat([anno_df_with_face, pd.DataFrame([anno_row])], ignore_index=True)
    return anno_df_with_face

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--anno_path', default=None, type=str) 
    parser.add_argument('--charbank_path', default=None, type=str)
    parser.add_argument('--video_dir', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--score_thresh', default=0.2, type=float, help='threshold on the cosine similarity for character recognition')
    parser.add_argument('--det_thresh', default=0.4, type=float, help='face detection threshold')
    parser.add_argument('--det_size', default=640, type=float, help='image size for face detection')
    parser.add_argument('--cache_dir', type=str, help='Path to save/load insightface model')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to extract from videos for face recognition')
    parser.add_argument('--first_n_samples', type=int, default=0, help='Only use first N samples')
    parser.add_argument('--query_bank_path', type=str, default=None,
                        help='Path to query bank .pkl (from build_query_bank.py). '
                             'When provided, uses clustering-calibrated centroids '
                             'instead of raw headshot embeddings.')
    args = parser.parse_args()

    # load query bank if provided
    query_bank = None
    if args.query_bank_path:
        query_bank = load_query_bank(args.query_bank_path)
        print(f"Loaded query bank from {args.query_bank_path} "
              f"({len(query_bank)} movies)")

    # initialise face detection model
    app = FaceAnalysis(name='buffalo_l', root=args.cache_dir)
    app.prepare(ctx_id=0, det_thresh=args.det_thresh, det_size=(args.det_size, args.det_size))  

    # load character bank
    TOP_K_CHARS = 10
    if args.dataset in ["cmdad", "tvad", "swissAD"]:
        with open(args.charbank_path) as fobj:
            charbank_dict = json.load(fobj)
    elif args.dataset == "madeval":
        with open(args.charbank_path, 'rb') as fobj:
            charbank_dict = pickle.load(fobj)
        charbank_dict = {str(key):value for key,value in charbank_dict.items()}
    else:
        print("Check dataset")
        sys.exit()

    # load AD annotaion file
    anno_df = pd.read_csv(args.anno_path)
    if args.first_n_samples>0:
        anno_df = anno_df.iloc[:args.first_n_samples]
    print(f"In total {len(anno_df)} AD annotations")
    anno_df["bboxes"] = [[[]] * args.num_frames] * len(anno_df)
    anno_df["pred_ids"] = [{}] * len(anno_df)

    # create a new file to save character recognition results
    anno_df_with_face = pd.DataFrame(columns=anno_df.columns)

    # save face bboxes and ids into annotation file
    anno_df_with_face = main(args, app, charbank_dict, anno_df, anno_df_with_face,
                             query_bank=query_bank)
    os.makedirs(args.output_dir, exist_ok=True)
    anno_df_with_face.to_csv(os.path.join(args.output_dir, f"{args.dataset}_anno_with_face_reproduced_{args.score_thresh}_{args.det_thresh}.csv"), index=False)
    

            





































