import cv2

def draw_box(img, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

def draw_text(img, text, pos, color=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

def draw_landmarks(img, landmarks):
    for (x, y) in landmarks.astype(int):
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
    return img

def draw_faces(img, faces, labels=None):
    """
    Draw bounding boxes, landmarks, and optional labels on detected faces.
    """
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = map(int, f.bbox)
        draw_box(img, (x1, y1, x2, y2))
        if f.kps is not None:
            draw_landmarks(img, f.kps)
        if labels:
            draw_text(img, labels[i], (x1, y1 - 10))
    return img