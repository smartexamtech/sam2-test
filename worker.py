import os
import cv2
import uuid
import boto3
from sqlalchemy import create_engine, Column, String, Integer, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. Database Setup
# ==========================================
DB_URL = f"mysql+pymysql://{os.getenv('DB_USER', 'root')}:{os.getenv('DB_PASS', 'root')}@{os.getenv('DB_HOST', 'localhost')}/{os.getenv('DB_NAME', 'studio_db')}"
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
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)
S3_BUCKET = os.getenv("AWS_S3_BUCKET")

# ==========================================
# 3. Simulation Logic
# ==========================================
def extract_character_frame(video_url, project_id):
    print(f"  ⬇️ Downloading video from S3...")
    s3_key = video_url.split(".amazonaws.com/")[1]
    temp_video = f"/tmp/{project_id}_temp.mp4"
    s3_client.download_file(S3_BUCKET, s3_key, temp_video)
    
    print(f"  🧠 Extracting frame (Simulating SAM 3)...")
    cap = cv2.VideoCapture(temp_video)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        temp_image = f"/tmp/{project_id}_char.png"
        cv2.imwrite(temp_image, frame)
        
        upload_key = f"projects/{project_id}/characters/char_01.png"
        print(f"  ⬆️ Uploading extracted frame to S3...")
        s3_client.upload_file(temp_image, S3_BUCKET, upload_key)
        
        os.remove(temp_video)
        os.remove(temp_image)
        
        return f"https://{S3_BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{upload_key}"
    return None

# ==========================================
# 4. The Batch Loop
# ==========================================
def process_queue():
    db = SessionLocal()
    try:
        # Find ALL queued projects
        queued_projects = db.query(Project).filter(Project.status == "QUEUED").all()
        
        if not queued_projects:
            print("No queued projects found. Exiting.")
            return

        print(f"Found {len(queued_projects)} project(s) in the queue.")
        
        for project in queued_projects:
            print(f"\nProcessing: {project.title} ({project.id})")
            
            # Update status to STARTED
            project.status = "EXTRACTION_STARTED"
            project.message = "Extracting characters..."
            db.commit()
            
            # Get the first video
            video = db.query(ReferenceVideo).filter(ReferenceVideo.project_id == project.id).first()
            if not video:
                project.status = "EXTRACTION_ERROR"
                project.message = "No reference video found."
                db.commit()
                continue
                
            # Run extraction
            try:
                s3_url = extract_character_frame(video.file_path, project.id)
                
                if s3_url:
                    new_char = ExtractedCharacter(
                        id=str(uuid.uuid4()),
                        project_id=project.id,
                        file_path=s3_url,
                        character_index=1
                    )
                    db.add(new_char)
                    
                    project.status = "EXTRACTION_COMPLETED"
                    project.message = "Extraction successful."
                    db.commit()
                    print(f"✅ Project {project.id} completed!")
                else:
                    project.status = "EXTRACTION_ERROR"
                    project.message = "Failed to read video frame."
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