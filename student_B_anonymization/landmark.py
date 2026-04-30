import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

def get_landmark_and_masked_image(image_path: str, output_size: int = 256):

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None, None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print(f"未检测到人脸: {image_path}")
        return None, None

    h, w = img.shape[:2]

    # landmark
    landmark_img = np.zeros((h, w, 3), dtype=np.uint8)
    for lm in results.multi_face_landmarks[0].landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(landmark_img, (x, y), 3, (255, 255, 255), -1)

    # masked_img
    mask = np.zeros((h, w), dtype=np.uint8)
    points = np.array([[int(lm.x * w), int(lm.y * h)] 
                       for lm in results.multi_face_landmarks[0].landmark])
    
    hull = cv2.convexHull(points)      
    cv2.fillConvexPoly(mask, hull, 255)    
    masked = img.copy()
    masked[mask > 0] = 0

    landmark_img = cv2.resize(landmark_img, (output_size, output_size))
    masked = cv2.resize(masked, (output_size, output_size))

    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

    return Image.fromarray(landmark_img), Image.fromarray(masked)