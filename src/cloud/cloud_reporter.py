import os
import json
import uuid
import firebase_admin
from firebase_admin import credentials, storage, db
from datetime import datetime

class CloudReporter:
    """
    Class to report injury detection results to the cloud (Firebase)
    """
    
    def __init__(self, config):
        """
        Initialize the cloud reporter with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.firebase_app = self._initialize_firebase()
        
        # Initialize Firebase Storage and Database if Firebase is initialized
        if self.firebase_app:
            try:
                # Use the default bucket or explicitly specify the bucket name
                self.storage_bucket = storage.bucket()
                
                # Print bucket name for debugging
                print(f"Using Firebase Storage bucket: {self.storage_bucket.name}")
                
                self.db_ref = db.reference('/injury_reports')
            except Exception as e:
                print(f"Error initializing storage bucket: {e}")
                self.storage_bucket = None
        
    def _initialize_firebase(self):
        """
        Initialize Firebase client
        
        Returns:
            Firebase app instance or None if initialization fails
        """
        try:
            # Check if Firebase is already initialized
            if len(firebase_admin._apps) > 0:
                return firebase_admin.get_app()
            
            # Get the path to the Firebase credentials file
            cred_path = self.config['cloud']['credentials_path']
            
            # Check if credentials file exists
            if not os.path.exists(cred_path):
                print(f"Warning: Firebase credentials file {cred_path} not found.")
                print("Cloud reporting will be disabled.")
                return None
            
            # Initialize Firebase with the credentials
            cred = credentials.Certificate(cred_path)
            
            # Get storage bucket from config - ensure correct format
            storage_bucket = self.config['cloud']['storage_bucket']
            if not storage_bucket.endswith('.appspot.com'):
                print("Warning: Storage bucket doesn't follow expected format. Using as is.")
            
            # Initialize the app with the credentials and database URL
            options = {
                'databaseURL': self.config['cloud']['database_url'],
                'storageBucket': storage_bucket  # Use the storage bucket from config
            }
            
            firebase_app = firebase_admin.initialize_app(cred, options)
            print(f"Firebase initialized successfully with storage bucket: {storage_bucket}")
            return firebase_app
            
        except Exception as e:
            print(f"Error initializing Firebase: {e}")
            print("Cloud reporting will be disabled.")
            return None
    
    def upload_report(self, detection_result, image_path=None):
        """
        Upload detection report to cloud
        
        Args:
            detection_result: Dictionary containing detection results
            image_path: Path to the image file (optional)
            
        Returns:
            dict: Upload result with status and report ID
        """
        if not self.firebase_app:
            return {
                "success": False,
                "message": "Firebase not initialized. Cloud reporting is disabled."
            }
        
        if not self.storage_bucket:
            return {
                "success": False,
                "message": "Firebase Storage bucket not initialized correctly."
            }
        
        try:
            # Generate a unique report ID
            report_id = str(uuid.uuid4())
            
            # Prepare the report data
            report_data = {
                "id": report_id,
                "timestamp": detection_result.get("timestamp", datetime.now().isoformat()),
                "injury_type": detection_result.get("injury_type", "unknown"),
                "confidence": detection_result.get("confidence", 0.0),
                "age": detection_result.get("age", "unknown"),
                "gender": detection_result.get("gender", "unknown"),
                "gender_confidence": detection_result.get("gender_confidence", 0.0),
                "created_at": datetime.now().isoformat()
            }
            
            # Upload the image to Firebase Storage if provided
            if image_path and os.path.exists(image_path):
                image_url = self._upload_image(image_path, report_id)
                if image_url:
                    report_data["image_url"] = image_url
            
            # Store the report in Firebase Realtime Database
            self.db_ref.child(report_id).set(report_data)
            
            print(f"Report uploaded successfully. Report ID: {report_id}")
            return {
                "success": True,
                "report_id": report_id,
                "message": "Report uploaded successfully."
            }
            
        except Exception as e:
            print(f"Error uploading report: {e}")
            return {
                "success": False,
                "message": f"Error uploading report: {str(e)}"
            }
    
    def _upload_image(self, image_path, report_id):
        """
        Upload an image to Firebase Storage
        
        Args:
            image_path: Path to the image file
            report_id: Report ID to use in the storage path
            
        Returns:
            str: Public URL of the uploaded image, or None if upload fails
        """
        try:
            # Get file extension
            _, ext = os.path.splitext(image_path)
            
            # Create a storage path
            storage_path = f"injury_images/{report_id}{ext}"
            
            print(f"Uploading image to path: {storage_path}")
            print(f"Using bucket: {self.storage_bucket.name}")
            
            # Upload the file
            blob = self.storage_bucket.blob(storage_path)
            blob.upload_from_filename(image_path)
            
            # Make the file publicly accessible
            blob.make_public()
            
            # Return the public URL
            return blob.public_url
            
        except Exception as e:
            print(f"Error uploading image: {e}")
            return None