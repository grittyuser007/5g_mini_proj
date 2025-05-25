import os
import cv2
import numpy as np
from datetime import datetime
import random

def generate_sample_images(output_dir="data", count=5):
    """
    Generate sample images for testing the YOLO-based injury detection system.
    
    Args:
        output_dir: Directory to save sample images
        count: Number of sample images to generate
        
    Returns:
        list: Paths to generated images
    """
    # Ensure output directory exists
    sample_dir = os.path.join(output_dir, "sample_images")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Define sample injury types
    injury_types = ["burn", "cut", "bruise", "fracture", "sprain", "head_injury"]
    
    # Generate sample images
    image_paths = []
    
    for i in range(count):
        # Create a base image that is more realistic for object detection
        img_size = (640, 480)
        # Create a more realistic background (skin tone or environment)
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * np.array([210, 180, 140], dtype=np.uint8)
        
        # Add a person-like shape to simulate a body part with injury
        # This will help YOLO detect a person or body part
        center_x = random.randint(img_size[0]//4, 3*img_size[0]//4)
        center_y = random.randint(img_size[1]//4, 3*img_size[1]//4)
        
        # Select a random injury type
        injury_type = random.choice(injury_types)
        
        # Draw a body part
        if random.random() < 0.6:  # 60% chance to draw a face-like shape
            # Draw a face-like oval
            face_size = random.randint(100, 200)
            cv2.ellipse(img, 
                       (center_x, center_y),
                       (face_size, int(face_size*1.3)),
                       0, 0, 360, (200, 200, 200), -1)
            
            # Add eyes
            eye_size = face_size // 10
            left_eye_x = center_x - face_size // 3
            right_eye_x = center_x + face_size // 3
            eyes_y = center_y - face_size // 6
            
            cv2.circle(img, (left_eye_x, eyes_y), eye_size, (70, 70, 70), -1)
            cv2.circle(img, (right_eye_x, eyes_y), eye_size, (70, 70, 70), -1)
            
            # Add mouth
            mouth_y = center_y + face_size // 3
            cv2.ellipse(img,
                       (center_x, mouth_y),
                       (face_size//3, face_size//6),
                       0, 0, 180, (70, 70, 70), -1)
            
            # Add injury based on type
            if injury_type == "head_injury":
                # Add a wound on the head
                injury_x = center_x + random.randint(-face_size//2, face_size//2)
                injury_y = center_y - face_size//2 - random.randint(-10, 20)
                cv2.circle(img, (injury_x, injury_y), face_size//6, (0, 0, 200), -1)
                
            elif injury_type == "cut":
                # Add a cut on the face
                start_x = center_x - face_size//4
                start_y = center_y - face_size//8
                end_x = center_x + face_size//4
                end_y = center_y + face_size//8
                cv2.line(img, (start_x, start_y), (end_x, end_y), (0, 0, 200), 3)
                
            elif injury_type == "bruise":
                # Add a bruise
                bruise_x = center_x - face_size//3
                bruise_y = center_y
                cv2.circle(img, (bruise_x, bruise_y), face_size//5, (140, 0, 140), -1)
        
        else:  # 40% chance to draw an arm or leg-like shape
            # Draw an arm or leg
            limb_width = random.randint(40, 80)
            limb_length = random.randint(200, 300)
            angle = random.randint(0, 180)
            
            # Create a rotated rectangle to simulate a limb
            rect = ((center_x, center_y), (limb_length, limb_width), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (200, 170, 140), -1)
            
            # Add injury based on type
            if injury_type == "fracture":
                # Create a jagged line to indicate fracture
                fx, fy = center_x, center_y
                points = []
                for j in range(6):
                    fx += random.randint(10, 20) * (1 if j % 2 == 0 else -1)
                    fy += random.randint(5, 15) * (1 if j % 2 == 0 else -1)
                    points.append((fx, fy))
                
                for j in range(len(points)-1):
                    cv2.line(img, points[j], points[j+1], (0, 0, 0), 2)
                    
            elif injury_type == "burn":
                # Create multiple red patches
                for _ in range(5):
                    bx = center_x + random.randint(-limb_length//3, limb_length//3)
                    by = center_y + random.randint(-limb_width//2, limb_width//2)
                    size = random.randint(10, 30)
                    cv2.circle(img, (bx, by), size, (0, 0, 200), -1)
                    
            elif injury_type == "sprain":
                # Create swelling effect
                cv2.circle(img, (center_x, center_y), limb_width, (150, 100, 230), -1)
        
        # Add text with injury type
        cv2.putText(img, f"Sample: {injury_type}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add text with random age and gender (for testing purposes)
        age = random.randint(18, 80)
        gender = random.choice(["Male", "Female"])
        
        cv2.putText(img, f"Age: {age}, Gender: {gender}", (10, img_size[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(sample_dir, f"sample_{injury_type}_{timestamp}_{i}.jpg")
        cv2.imwrite(img_path, img)
        image_paths.append(img_path)
        
        print(f"Generated sample image: {img_path}")
    
    return image_paths

def generate_sample_video(output_dir="data", duration=10, fps=30):
    """
    Generate a sample video for testing the YOLO-based injury detection system.
    
    Args:
        output_dir: Directory to save sample video
        duration: Duration of video in seconds
        fps: Frames per second
        
    Returns:
        str: Path to generated video
    """
    # Ensure output directory exists
    sample_dir = os.path.join(output_dir, "sample_videos")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Video properties
    img_size = (640, 480)
    total_frames = duration * fps
    
    # Define injury types to cycle through
    injury_types = ["burn", "cut", "bruise", "fracture", "head_injury", "no_injury"]
    
    # Random age and gender for this "person"
    age = random.randint(18, 80)
    gender = random.choice(["Male", "Female"])
    
    # Create video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(sample_dir, f"sample_injuries_video_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
    video = cv2.VideoWriter(video_path, fourcc, fps, img_size)
    
    if not video.isOpened():
        print("Error: Could not create video file.")
        return None
    
    # Generate frames
    for frame_idx in range(total_frames):
        # Create base frame - simulated person
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * np.array([210, 180, 140], dtype=np.uint8)
        
        # Change injury type every 2 seconds
        current_injury = injury_types[(frame_idx // (fps * 2)) % len(injury_types)]
        
        # Calculate animation progress within the 2-second window
        progress = (frame_idx % (fps * 2)) / (fps * 2)
        
        # Draw a person-like shape (simplified for YOLO detection)
        # Draw a face
        face_size = 120
        cv2.ellipse(img, 
                   (320, 180),
                   (face_size, int(face_size*1.3)),
                   0, 0, 360, (200, 200, 200), -1)
        
        # Add eyes
        eye_size = face_size // 10
        cv2.circle(img, (320 - face_size//3, 180 - face_size//6), eye_size, (70, 70, 70), -1)
        cv2.circle(img, (320 + face_size//3, 180 - face_size//6), eye_size, (70, 70, 70), -1)
        
        # Add mouth
        cv2.ellipse(img,
                   (320, 180 + face_size//3),
                   (face_size//3, face_size//6),
                   0, 0, 180, (70, 70, 70), -1)
        
        # Add body
        cv2.rectangle(img, (280, 230), (360, 400), (200, 170, 140), -1)
        
        # Add arms
        cv2.rectangle(img, (230, 250), (280, 280), (200, 170, 140), -1)  # Left arm
        cv2.rectangle(img, (360, 250), (410, 280), (200, 170, 140), -1)  # Right arm
        
        # Add legs
        cv2.rectangle(img, (290, 400), (320, 470), (200, 170, 140), -1)  # Left leg
        cv2.rectangle(img, (321, 400), (350, 470), (200, 170, 140), -1)  # Right leg
        
        # Add injuries based on type
        if current_injury != "no_injury":
            if current_injury == "burn":
                # Burns that grow in size
                size = int(20 + 30 * progress)
                for i in range(3):
                    x = 370 + i * 15
                    y = 260 + i * 10
                    cv2.circle(img, (x, y), size, (50, 50, 200), -1)
                    
            elif current_injury == "cut":
                # Cut that extends
                length = int(20 + 80 * progress)
                cv2.line(img, (290, 300), (290 + length, 330), (50, 50, 200), 3)
                
            elif current_injury == "bruise":
                # Bruise that darkens
                intensity = int(50 + 150 * progress)
                cv2.circle(img, (240, 260), 30, (intensity, 50, intensity), -1)
                
            elif current_injury == "fracture":
                # Fracture line on leg
                cv2.line(img, (300, 420), (310, 430), (0, 0, 0), 2)
                cv2.line(img, (310, 430), (300, 440), (0, 0, 0), 2)
                cv2.line(img, (300, 440), (310, 450), (0, 0, 0), 2)
                
            elif current_injury == "head_injury":
                # Head injury that spreads
                size = int(10 + 40 * progress)
                cv2.circle(img, (350, 150), size, (50, 50, 200), -1)
        
        # Add text with injury type, age and gender
        cv2.putText(img, f"Injury: {current_injury}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, f"Age: {age}, Gender: {gender}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add frame to video
        video.write(img)
    
    # Release resources
    video.release()
    print(f"Generated sample video: {video_path}")
    
    return video_path

if __name__ == "__main__":
    print("Generating sample data for YOLO-based injury detection testing...")
    generate_sample_images(count=10)
    generate_sample_video(duration=10)
    print("Sample data generation complete.")