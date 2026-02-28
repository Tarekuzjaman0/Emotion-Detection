import cv2
from deepface import DeepFace

# Iriun camera index (use 1 or 2 if 0 doesn't work)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream. Try '1' or '2' if '0' doesn't work.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
        
    # Resize the frame slightly for faster processing
    frame = cv2.resize(frame, (640, 480))
    
    try:
        # Detect face and emotion using DeepFace
        # enforce_detection=False prevents the program from crashing if no face is found
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # DeepFace returns results as a list
        for face in results:
            # Coordinates for the bounding box around the face
            x = face['region']['x']
            y = face['region']['y']
            w = face['region']['w']
            h = face['region']['h']
            
            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract the dominant emotion
            dominant_emotion = face['dominant_emotion']
            
            # Display the emotion name on the screen
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    except Exception as e:
        # Pass if no face is detected or an error occurs
        pass

    # Display the video window
    cv2.imshow('Emotion Detection - DeepFace & Iriun', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()