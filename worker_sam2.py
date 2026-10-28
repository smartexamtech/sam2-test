import os
import cv2
import uuid
import boto3
import torch
import numpy as np
import subprocess
from sqlalchemy import create_engine, Column, String, Integer, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
from sam2.build_sam import build_sam2_video_predictor

load_dotenv()

# ==========================================
# 1. Database Setup
# ==========================================
DB_URL = f"mysql+pymysql://{os.getenv('DB_USER', 'admin')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME', 'studio_db')}"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Project(Base):
    __tablename__ = "projects"
    id = Column(String(36), primary_key=True)
    title = Column(String(255))
    status = Column(String(50))
    message = Column(String(255))

class ReferenceVideo(Base):
    __tablename__ = "reference_videos"
    id = Column(String(36), primary_key=True)
    project_id = Column(String(36), ForeignKey("projects.id"))
    file_path = Column(String(1000))

class ExtractedCharacter(Base):
    __tablename__ = "extracted_characters"
    id = Column(String(36), primary_key=True)
    project_id = Column(String(36), ForeignKey("projects.id"))
    file_path = Column(String(1000))
    character_index = Column(Integer)

# ==========================================
# 2. S3 Setup
# ==========================================
s3_client = boto3.client(
    's3',
    region_name=os.getenv("AWS_REGION", "ap-south-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)
S3_BUCKET = os.getenv("AWS_S3_BUCKET")

# ==========================================
# 3. Real SAM 2 Logic
# ==========================================
def extract_characters_real(video_url, project_id):
    print(f"  ⬇️ Downloading video from S3...")
    s3_key = video_url.split(".amazonaws.com/")[1]
    temp_video = f"/tmp/{project_id}_temp.mp4"
    s3_client.download_file(S3_BUCKET, s3_key, temp_video)

    print("Loading SAM 2 Model (GPU Required)...")
    checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise Exception("SAM 2 requires a GPU. Please run on AWS instance.")

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    # Init video state
    inference_state = predictor.init_state(video_path=temp_video)

    # Prompt the center of the first frame
    cap = cv2.VideoCapture(temp_video)
    ret, first_frame = cap.read()
    cap.release()
    
    if not ret:
        return None

    h, w, _ = first_frame.shape
    center_point = np.array([[w // 2, h // 2]])
    point_labels = np.array([1]) # 1 = foreground

    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=center_point,
        labels=point_labels,
    )

    print("Propagating mask across video...")
    uploaded_urls = []
    frame_count = 0

    with torch.no_grad():
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            # Only process every 30th frame to save time/storage
            if frame_count % 30 != 0:
                frame_count += 1
                continue

            cap = cv2.VideoCapture(temp_video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, out_frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if not ret: continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255

            # Create RGBA (Transparent Background)
            rgba = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = frame_rgb
            rgba[..., 3] = mask

            temp_image = f"/tmp/{project_id}_char_{frame_count}.png"
            cv2.imwrite(temp_image, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

            upload_key = f"projects/{project_id}/characters/char_{frame_count:03d}.png"
            s3_client.upload_file(temp_image, S3_BUCKET, upload_key)
            
            url = f"https://{S3_BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{upload_key}"
            uploaded_urls.append(url)
            
            os.remove(temp_image)
            frame_count += 1

    os.remove(temp_video)
    return uploaded_urls

# ==========================================
# 4. The Batch Loop
# ==========================================
def process_queue():
    db = SessionLocal()
    try:
        queued_projects = db.query(Project).filter(Project.status == "QUEUED").all()
        
        if not queued_projects:
            print("No queued projects found. Exiting.")
            return

        print(f"Found {len(queued_projects)} project(s) in the queue.")
        
        for project in queued_projects:
            print(f"\nProcessing: {project.title} ({project.id})")
            
            project.status = "EXTRACTION_STARTED"
            project.message = "SAM 2 is extracting characters..."
            db.commit()
            
            video = db.query(ReferenceVideo).filter(ReferenceVideo.project_id == project.id).first()
            if not video:
                project.status = "EXTRACTION_ERROR"
                project.message = "No reference video found."
                db.commit()
                continue
                
            try:
                urls = extract_characters_real(video.file_path, project.id)
                
                if urls:
                    for i, url in enumerate(urls):
                        new_char = ExtractedCharacter(
                            id=str(uuid.uuid4()),
                            project_id=project.id,
                            file_path=url,
                            character_index=i + 1
                        )
                        db.add(new_char)
                    
                    project.status = "EXTRACTION_COMPLETED"
                    project.message = f"Extracted {len(urls)} character frames."
                    db.commit()
                    print(f"✅ Project {project.id} completed!")
                else:
                    project.status = "EXTRACTION_ERROR"
                    project.message = "SAM 2 failed to read video."
                    db.commit()
            except Exception as e:
                print(f"❌ Error: {e}")
                project.status = "EXTRACTION_ERROR"
                project.message = str(e)
                db.commit()

    finally:
        db.close()
        print("\nBatch processing finished.")

if __name__ == "__main__":
    process_queue()
    
    # Auto-shutdown the instance
    print("\n🛑 Shutting down the GPU instance to save money...")
    subprocess.run(["sudo", "shutdown", "-h", "now"])