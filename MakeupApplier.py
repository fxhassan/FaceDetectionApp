from PIL import Image, ImageDraw
import face_recognition

image = face_recognition.load_image_file("face.jpg")

face_landmarks_list = face_recognition.face_landmarks(image)

pil_image = Image.fromarray(image)

d = ImageDraw.Draw(pil_image, 'RGBA')

for face_landmarks in face_landmarks_list:
    # The face landmark detection model returns these features:
    #  - chin, left_eyebrow, right_eyebrow, nose_bridge, nose_tip, left_eye, right_eye, top_lip, bottom_lip

    # Draw a line over the eyebrows
    d.line(face_landmarks["left_eyebrow"], fill=(128, 0, 128, 0), width=5)
    d.line(face_landmarks["right_eyebrow"], fill=(128, 0, 128, 0), width=5)
    d.polygon(face_landmarks["top_lip"], fill=(128, 60, 128, 100))
    d.polygon(face_landmarks["bottom_lip"], fill=(18, 60, 128, 100))

pil_image.show()
