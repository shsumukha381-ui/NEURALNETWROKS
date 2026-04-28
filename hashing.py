
import hashlib
import json
import os
import threading
from datetime import datetime
from pathlib import Path


class EvidenceHasher:
  
    
    def __init__(self, log_file="security_log.json"):
        """
        Initialize evidence hasher
        
        Args:
            log_file: Path to security log JSON file
        """
        self.log_file = log_file
        self.lock = threading.Lock()
        
        # Create log file if it doesn't exist
        if not os.path.exists(log_file):
            self._initialize_log()
    
    def _initialize_log(self):
        """Initialize empty security log"""
        log_data = {
            "system": "SafetyNet AI - Evidence Integrity System",
            "hash_algorithm": "SHA-256",
            "created": datetime.now().isoformat(),
            "incidents": []
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"✓ Security log initialized: {self.log_file}")
    
    def calculate_sha256(self, file_path, chunk_size=8192):
        """
        Calculate SHA-256 hash of a file
        
        Args:
            file_path: Path to file to hash
            chunk_size: Size of chunks to read (8KB default for efficiency)
            
        Returns:
            hex_digest: SHA-256 hash as hexadecimal string
        """
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large videos efficiently
                while chunk := f.read(chunk_size):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
        
        except Exception as e:
            print(f"❌ Error calculating hash: {e}")
            return None
    
    def hash_evidence_async(self, video_path, incident_info=None):
        """
        Hash video evidence in background thread (non-blocking)
        
        Args:
            video_path: Path to video file
            incident_info: Dictionary with incident details
        """
        thread = threading.Thread(
            target=self._hash_and_log,
            args=(video_path, incident_info),
            daemon=True
        )
        thread.start()
        print(f"🔐 Hashing evidence in background: {os.path.basename(video_path)}")
    
    def _hash_and_log(self, video_path, incident_info):
        """
        Internal method to hash file and log to security log
        
        Args:
            video_path: Path to video file (must be permanent location)
            incident_info: Dictionary with incident details (including anomaly_type)
        """
        try:
            # CRITICAL: Verify file exists before hashing
            if not os.path.exists(video_path):
                print(f"❌ Error: Video file not found at: {video_path}")
                print(f"   Cannot hash non-existent file!")
                return
            
            # Verify file is not empty
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                print(f"❌ Error: Video file is empty (0 bytes): {video_path}")
                return
            
            print(f"✓ File verified: {os.path.basename(video_path)} ({file_size / (1024*1024):.2f} MB)")
            
            # Wait a moment to ensure file is fully written and closed
            import time
            time.sleep(0.5)
            
            # Calculate hash
            print(f"🔐 Calculating SHA-256 hash...")
            file_hash = self.calculate_sha256(video_path)
            
            if file_hash is None:
                print(f"❌ Error: Failed to calculate hash for {video_path}")
                return
            
            # Get file metadata
            file_name = os.path.basename(video_path)
            
            # Create incident record
            incident_record = {
                "timestamp": datetime.now().isoformat(),
                "filename": file_name,
                "filepath": os.path.abspath(video_path),
                "sha256_hash": file_hash,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "anomaly_type": incident_info.get("anomaly_type", "Unknown")  # Multi-class support
            }
            
            # Add incident info if provided
            if incident_info:
                incident_record.update(incident_info)
            
            # Append to security log (thread-safe)
            with self.lock:
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
                
                log_data["incidents"].append(incident_record)
                log_data["last_updated"] = datetime.now().isoformat()
                log_data["total_incidents"] = len(log_data["incidents"])
                
                with open(self.log_file, 'w') as f:
                    json.dump(log_data, f, indent=2)
            
            print(f"✓ Evidence hashed and logged successfully!")
            print(f"  Filename: {file_name}")
            print(f"  Anomaly Type: {incident_record['anomaly_type']}")
            print(f"  File Size: {incident_record['file_size_mb']} MB")
            print(f"  SHA-256: {file_hash[:32]}...{file_hash[-32:]}")
            print(f"  Permanent Path: {os.path.abspath(video_path)}")
            print(f"  Security Log: {os.path.abspath(self.log_file)}")
        
        except FileNotFoundError as e:
            print(f"❌ FileNotFoundError: {e}")
            print(f"   The video file must exist before hashing!")
            print(f"   Expected path: {video_path}")
        except Exception as e:
            print(f"❌ Error in hash_and_log: {e}")
            import traceback
            traceback.print_exc()
    
    def verify_evidence(self, video_path):
        """
        Verify video integrity by comparing current hash with logged hash
        
        Args:
            video_path: Path to video file to verify
            
        Returns:
            tuple: (is_valid, message, original_record)
        """
        file_name = os.path.basename(video_path)
        
        # Calculate current hash
        current_hash = self.calculate_sha256(video_path)
        
        if current_hash is None:
            return False, "Error calculating hash", None
        
        # Find original record in log
        try:
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
            
            # Search for matching filename
            original_record = None
            for incident in log_data["incidents"]:
                if incident["filename"] == file_name:
                    original_record = incident
                    break
            
            if original_record is None:
                return False, f"No record found for {file_name}", None
            
            # Compare hashes
            original_hash = original_record["sha256_hash"]
            
            if current_hash == original_hash:
                return True, "✓ VERIFIED: Video integrity intact", original_record
            else:
                return False, "❌ TAMPERED: Hash mismatch detected!", original_record
        
        except Exception as e:
            return False, f"Error reading log: {e}", None
    
    def get_incident_summary(self):
        """
        Get summary of all logged incidents
        
        Returns:
            dict: Summary statistics
        """
        try:
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
            
            total_incidents = len(log_data["incidents"])
            total_size_mb = sum(inc["file_size_mb"] for inc in log_data["incidents"])
            
            return {
                "total_incidents": total_incidents,
                "total_size_mb": round(total_size_mb, 2),
                "log_created": log_data.get("created"),
                "last_updated": log_data.get("last_updated")
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def export_chain_of_custody(self, output_file="chain_of_custody.txt"):
        """
        Export chain of custody report for legal purposes
        
        Args:
            output_file: Path to output text file
        """
        try:
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
            
            with open(output_file, 'w') as f:
                f.write("="*70 + "\n")
                f.write("SAFETYNET AI - CHAIN OF CUSTODY REPORT\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"System: {log_data['system']}\n")
                f.write(f"Hash Algorithm: {log_data['hash_algorithm']}\n")
                f.write(f"Log Created: {log_data['created']}\n")
                f.write(f"Last Updated: {log_data.get('last_updated', 'N/A')}\n")
                f.write(f"Total Incidents: {len(log_data['incidents'])}\n\n")
                
                f.write("="*70 + "\n")
                f.write("INCIDENT RECORDS\n")
                f.write("="*70 + "\n\n")
                
                for i, incident in enumerate(log_data["incidents"], 1):
                    f.write(f"INCIDENT #{i}\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Timestamp: {incident['timestamp']}\n")
                    f.write(f"Filename: {incident['filename']}\n")
                    f.write(f"File Path: {incident['filepath']}\n")
                    f.write(f"File Size: {incident['file_size_mb']} MB\n")
                    f.write(f"SHA-256 Hash:\n  {incident['sha256_hash']}\n")
                    
                    # Add additional incident info
                    for key, value in incident.items():
                        if key not in ['timestamp', 'filename', 'filepath', 'sha256_hash', 'file_size_bytes', 'file_size_mb']:
                            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                    
                    f.write("\n")
                
                f.write("="*70 + "\n")
                f.write("END OF REPORT\n")
                f.write("="*70 + "\n")
            
            print(f"✓ Chain of custody exported: {output_file}")
            return True
        
        except Exception as e:
            print(f"❌ Error exporting chain of custody: {e}")
            return False


def test_evidence_hasher():
    """Test the evidence hasher with a sample file"""
    print("\n" + "="*70)
    print("Testing Evidence Hasher")
    print("="*70 + "\n")
    
    hasher = EvidenceHasher("test_security_log.json")
    
    # Test with this script file
    test_file = __file__
    
    print(f"Testing with file: {test_file}\n")
    
    # Calculate hash
    print("1. Calculating SHA-256 hash...")
    file_hash = hasher.calculate_sha256(test_file)
    print(f"   Hash: {file_hash}\n")
    
    # Log evidence
    print("2. Logging evidence...")
    incident_info = {
        "mode": "TEST",
        "risk_score": 0.95,
        "anomaly_type": "Test Incident"
    }
    hasher._hash_and_log(test_file, incident_info)
    print()
    
    # Verify evidence
    print("3. Verifying evidence integrity...")
    is_valid, message, record = hasher.verify_evidence(test_file)
    print(f"   {message}\n")
    
    # Get summary
    print("4. Getting incident summary...")
    summary = hasher.get_incident_summary()
    print(f"   Total Incidents: {summary['total_incidents']}")
    print(f"   Total Size: {summary['total_size_mb']} MB\n")
    
    # Export chain of custody
    print("5. Exporting chain of custody...")
    hasher.export_chain_of_custody("test_chain_of_custody.txt")
    print()
    
    print("="*70)
    print("Test Complete!")
    print("="*70)


if __name__ == "__main__":
    test_evidence_hasher()
