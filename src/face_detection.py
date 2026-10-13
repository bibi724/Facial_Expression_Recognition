import cv2
import dlib
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
image_path = project_dir / "images" / "demo.jpg"

img = cv2.imread(str(image_path))

if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

detector = dlib.get_frontal_face_detector()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = detector(gray, 1)

print(f"Faces detected: {len(faces)}")

for i, face in enumerate(faces):
    x = max(0, face.left())
    y = max(0, face.top())
    x2 = min(gray.shape[1], face.right())
    y2 = min(gray.shape[0], face.bottom())

    face_crop = gray[y:y2, x:x2]

    if face_crop.size == 0:
        continue

    face_48 = cv2.resize(face_crop, (48, 48))

    face_display = cv2.resize(
        face_48,
        (288, 288),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow(f"Face {i + 1} - 48x48", face_display)

    cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)

cv2.imshow("Face Detection - Dlib", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
