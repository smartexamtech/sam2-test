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
# Load the Video Predictor (not the Image Predictor)
predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=device)

def process_video_full(video_path, output_dir, frame_skip=10):
    """
    frame_skip: Process every Nth frame. 
                frame_skip=1 -> every frame (slowest, best quality)
                frame_skip=10 -> every 10th frame (faster)
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    chars_dir = os.path.join(output_dir, video_name, "characters")
    temp_frames_dir = os.path.join(output_dir, video_name, "temp_frames")
    os.makedirs(chars_dir, exist_ok=True)
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    print(f"🎬 Processing {video_name}...")
    
    # 1. Extract ALL frames (or every Nth frame) to a temp folder for SAM 2 to read
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
    
    # 2. Tell SAM 2 to load the frames into its internal memory
    inference_state = predictor.init_state(video_path=temp_frames_dir)
    
    # 3. On the FIRST frame, click the center to identify the character
    first_frame_path = os.path.join(temp_frames_dir, "00000.jpg")
    first_frame = cv2.imread(first_frame_path)
    h, w, _ = first_frame.shape
    
    # Prompt: Click center of the image (assumes character is roughly in middle)
    center_point = np.array([[w // 2, h // 2]])
    point_labels = np.array([1]) # 1 = foreground
    
    # Add the click prompt to frame 0
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1, # Object ID for our character
        points=center_point,
        labels=point_labels,
    )
    
    # 4. Propagate the mask through the entire video!
    print(f"  🧠 Tracking character across all {saved_count} frames...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # 5. Save the cropped character for every Nth frame (to avoid thousands of images)
    save_every = 5 # Save 1 image every 5 frames (adjust based on video length)
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
            # Apply mask
            character_cropped = frame_rgb * mask[..., None]
            
            char_path = os.path.join(chars_dir, f"char_{frame_idx:05d}.png")
            cv2.imwrite(char_path, cv2.cvtColor(character_cropped, cv2.COLOR_RGB2BGR))
            saved_images += 1
            
    print(f"  ✅ Saved {saved_images} character images to {chars_dir}")
    
    # 6. Clean up temp frames to save disk space
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
            frame_skip=10  # Process every 10th frame
        )
        
    print("🚀 All videos processed!")