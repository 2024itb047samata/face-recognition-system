import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Load known images
vk_image = face_recognition.load_image_file("photos/vk1.jpg")
vk_encoding = face_recognition.face_encodings(vk_image)[0]

mr_image = face_recognition.load_image_file("photos/mr2.jpeg")
mr_encoding = face_recognition.face_encodings(mr_image)[0]

kk_image = face_recognition.load_image_file("photos/KK1.jpeg")
kk_encoding = face_recognition.face_encodings(kk_image)[0]

carry_image = face_recognition.load_image_file("photos/carry2.jpeg")
carry_encoding = face_recognition.face_encodings(carry_image)[0]

as_image = face_recognition.load_image_file("photos/AS1.jpeg")
as_encoding = face_recognition.face_encodings(as_image)[0]

ab_image = face_recognition.load_image_file("photos/alia-bhatt2.jpg")
ab_encoding = face_recognition.face_encodings(ab_image)[0]

known_face_encodings = [
    vk_encoding, mr_encoding, kk_encoding,
    carry_encoding, as_encoding, ab_encoding
]

known_face_names = [
    "Virat", "Jimmy", "Katrina",
    "Ajey", "Anushka", "Alia"
]

students = known_face_names.copy()

# CSV setup
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + ".csv", "w", newline="")
lnwriter = csv.writer(f)

# Read test image
frame = cv2.imread("photos/vk1.jpg")
rgb_frame = frame[:, :, ::-1]

face_locations = face_recognition.face_locations(rgb_frame)
face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(
        known_face_encodings, face_encoding
    )

    name = "Unknown"
    distances = face_recognition.face_distance(
        known_face_encodings, face_encoding
    )

    best_match_index = np.argmin(distances)

    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    if name in students:
        students.remove(name)
        current_time = now.strftime("%H-%M-%S")
        lnwriter.writerow([name, "Present", current_time])

# Show image
cv2.imshow("Output", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

f.close()
