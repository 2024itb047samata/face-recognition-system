import os
from import_face_recognition import predict  # adjust if function name is different

test_folder = "test"

for img in os.listdir(test_folder):
    path = os.path.join(test_folder, img)
    
    result = predict(path)   # your function
    
    print(f"{img} -> {result}")
