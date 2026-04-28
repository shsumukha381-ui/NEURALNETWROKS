1. Hardware & StabilityResolved an initial 0xC0000005 memory violation by implementing DirectShow (cv2.CAP_DSHOW) for the camera interface. Successfully established a stable 30 FPS real-time feed by optimizing MediaPipe delegates for the RTX 3050 GPU.2. Data & Feature EngineeringDeveloped a custom dataset of 80 motion sequences (2,400+ frames). Engineered a 120-dimensional feature vector per frame that calculates joint velocity and Euclidean distances ($d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$), providing the AI with a mathematical model of kinetic intent.3. AI Training & ArchitectureImplemented a Bidirectional LSTM with an Attention Mechanism for temporal analysis. The model achieved 100% validation accuracy within 200 epochs, effectively minimizing Binary Cross-Entropy loss to isolate aggressive behavioral signatures.4. Security & Evidence IntegrityBuilt a Digital Chain of Custody backend. Upon detection, the system triggers background recording and generates a SHA-256 cryptographic hash of the footage. This ensures all evidence is tamper-proof and verifiable in the security_log.json.5. Status & Next StepsThe core detection and alerting pipeline is fully functional. We are now transitioning to Multi-Class Classification to distinguish between specific anomalies, such as physical altercations and medical emergencies (falls).
   
1. Multi-Class Logic Upgrade
The system has been upgraded from binary (Anomaly/Normal) to Multi-Class Classification. By implementing a Softmax output layer, the model now specifically identifies kinetic signatures for Fighting (security threat) and Falling (medical emergency), rather than just flagging generic "bad" behavior.

2. Permanent Evidence Storage
We resolved a file-handling issue where Gradio was storing footage in temporary directories. We implemented a robust storage logic using the shutil library to move processed clips into a permanent recordings/ folder on the local SSD. This ensures that the SHA-256 hash recorded in the logs points to a stable, verifiable file.

3. UI & Logging Refinement
The Gradio Dashboard now acts as a full forensic suite. It displays the real-time feed, the calculated risk probability for each specific anomaly, the cryptographic SHA-256 hash, and the AI-generated narrative summary side-by-side.


4. Scaling with Transfer Learning
Integrated 1,000 images from the UCF dataset into the training pipeline. Replaced the baseline model with a ResNet18 backbone, leveraging pre-trained ImageNet weights to accelerate feature extraction and improve detection accuracy on the RTX 3050.

5. Multi-Class Optimization
Transitioned from binary detection to a 3-class system (Fighting, Falling, Normal). Resolved a critical AttributeError by correctly mapping model.fc.in_features to the ResNet architecture, ensuring the neural network "head" aligns with the 512-dimensional feature vector.

6. Permanent Evidence Storage
Fixed a critical data loss issue where Gradio was storing videos in temporary folders. Implemented a Permanent I/O Bridge using shutil to move incident clips into the local recordings/ directory, ensuring evidence survives application restarts.

7. Cryptographic Integration
Synchronized the SHA-256 hashing logic with the new storage path. Every anomaly is now captured as a 10-second clip, moved to the local vault, and cryptographically signed. The resulting hash is logged in security_log.json, creating a tamper-proof digital chain of custody.
