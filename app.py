import cv2
from deepface import DeepFace

# Iriun ক্যামেরার ইনডেক্স (0 কাজ না করলে 1 বা 2 দিবেন)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream. '0' কাজ না করলে '1' বা '2' দিয়ে চেষ্টা করুন।")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
        
    # ফ্রেম সাইজ কিছুটা ছোট করে নেওয়া
    frame = cv2.resize(frame, (640, 480))
    
    try:
        # DeepFace দিয়ে ফেস এবং ইমোশন ডিটেক্ট করা
        # enforce_detection=False দেওয়া হয়েছে যাতে ফেস না পেলেও প্রোগ্রাম ক্র্যাশ না করে
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # DeepFace এর রেজাল্ট একটি লিস্ট হিসেবে আসে
        for face in results:
            # মুখের চারপাশের বক্স এর কোঅর্ডিনেট
            x = face['region']['x']
            y = face['region']['y']
            w = face['region']['w']
            h = face['region']['h']
            
            # বক্স ড্র করা
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # সবচেয়ে বেশি মিলে যাওয়া ইমোশন বের করা
            dominant_emotion = face['dominant_emotion']
            
            # স্ক্রিনে ইমোশন এর নাম দেখানো
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    except Exception as e:
        # কোনো কারণে ফেস বুঝতে না পারলে পাস করবে
        pass

    # ভিডিও উইন্ডো দেখানো
    cv2.imshow('Emotion Detection - DeepFace & Iriun', frame)

    # 'q' চাপলে বের হয়ে যাবে
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ক্যামেরা রিলিজ করা
cap.release()
cv2.destroyAllWindows()