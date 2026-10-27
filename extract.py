import os
import cv2
import torch
import numpy as np
import glob

from sam2.build_sam import build_sam2_video_predictor

# Paths
VIDEO_DIR = "/app/videos"
OUTPUT_DIR = "/app/output"

# Config
CHECKPOINT = "/app/checkpoints/sam2.1_hiera_small.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔧 Using device: {device}")
predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=device)

def process_video_full(video_path, output_dir, frame_skip=10):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    chars_dir = os.path.join(output_dir, video_name, "characters")
    temp_frames_dir = os.path.join(output_dir, video_name, "temp_frames")
    os.makedirs(chars_dir, exist_ok=True)
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    print(f"🎬 Processing {video_name}...")
    
    # 1. Extract frames
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_count % frame_skip == 0:
            temp_path = os.path.join(temp_frames_dir, f"{saved_count:05d}.jpg")
            cv2.imwrite(temp_path, frame)
            saved_count += 1
            
        frame_count += 1
    cap.release()
    
    print(f"  📸 Extracted {saved_count} frames for SAM 2 processing.")
    
    # 2. Load frames into SAM 2
    inference_state = predictor.init_state(video_path=temp_frames_dir)
    
    # 3. Prompt the first frame (click center)
    first_frame_path = os.path.join(temp_frames_dir, "00000.jpg")
    first_frame = cv2.imread(first_frame_path)
    h, w, _ = first_frame.shape
    
    center_point = np.array([[w // 2, h // 2]])
    point_labels = np.array([1])
    
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=center_point,
        labels=point_labels,
    )
    
    # 4. Propagate through all frames
    print(f"  🧠 Tracking character across all {saved_count} frames...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # 5. Save extracted characters with transparent background
    save_every = 5
    saved_images = 0
    
    for frame_idx, segments in video_segments.items():
        if frame_idx % save_every != 0:
            continue
            
        frame_path = os.path.join(temp_frames_dir, f"{frame_idx:05d}.jpg")
        if not os.path.exists(frame_path):
            continue
            
        frame = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for obj_id, mask in segments.items():
            # Create RGBA image (transparent background)
            rgba = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = frame_rgb
            rgba[..., 3] = (mask * 255).astype(np.uint8)
            
            char_path = os.path.join(chars_dir, f"char_{frame_idx:05d}.png")
            cv2.imwrite(char_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            saved_images += 1
            
    print(f"  ✅ Saved {saved_images} character images to {chars_dir}")
    
    # 6. Cleanup
    for f in glob.glob(os.path.join(temp_frames_dir, "*.jpg")):
        os.remove(f)
    os.rmdir(temp_frames_dir)
    
    print(f"✅ Finished {video_name}.\n")

if __name__ == "__main__":
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    print(f"📁 Found {len(videos)} videos.")
    
    for video in videos:
        process_video_full(
            os.path.join(VIDEO_DIR, video), 
            OUTPUT_DIR, 
            frame_skip=10
        )
        
    print("🚀 All videos processed!")