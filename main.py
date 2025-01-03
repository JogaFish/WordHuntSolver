import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


# load yolo model
yolo = YOLO("predict_model.pt")

# load OCR reader model
# reader = easyocr.Reader(['en'])

# this is using yolo prediction to show the frames, used for testing
# results = yolo.predict(source=0, show=True)

# Load the video capture
videoCap = cv2.VideoCapture(0)


# function to obtain the colours of the bounded boxes, dependent on the class number
def getColours(cls_num):
    if cls_num == 0:
        return 255, 0, 0        # Blue for boxes
    elif cls_num == 1:
        return 0, 255, 0        # Green for letterboxes
    else:
        return 0, 0, 0    # Return gray for unknown class number input


# Example model (simple CNN)
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # For grayscale (1 channel)
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(16 * 14 * 14, 26)  # Output size = 26 (letters A-Z)
)

# Reinitialize the model
model.load_state_dict(torch.load("ocr_model.pth"))
model.eval()
print("Model loaded successfully")

transform = transforms.Compose([
    transforms.Grayscale(),                # Convert to grayscale
    transforms.Resize((28, 28)),           # Resize to match input size
    transforms.ToTensor(),                 # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize with mean and std
])
alphabet = "AEHINOPRST"

image_counter = 0

while True:
    ret, frame = videoCap.read()
    original_frame = frame
    if not ret:
        continue

    results = yolo.track(frame, stream=True)
    letters = []

    for result in results:
        # get the classes names
        classes_names = result.names

        # iterate over each box
        for box in result.boxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.80:
                # get coordinates and convert to integers
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class, name and colour
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                colour = getColours(cls)

                if cls == 1:
                    # Crop the detected object
                    cropped_object = original_frame[y1:y2, x1:x2]

                    # Save the cropped image
                    if cropped_object.size > 0:                 # Check to ensure cropping was successful
                        img_path = f'temp_img_{image_counter}.png'
                        cv2.imwrite(img_path, cropped_object)
                        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_object, cv2.COLOR_BGR2RGB))

                        # Preprocess and predict
                        input_tensor = transform(cropped_pil).unsqueeze(0)
                        output = model(input_tensor)
                        _, predicted_class = torch.max(output, 1)

                        # Get the letter
                        predicted_letter = alphabet[predicted_class.item()]
                        print(f"{predicted_letter}")

                        image_counter += 1
                        # draw the rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                        # put the letter on the image
                        cv2.putText(frame, f'{predicted_letter}', (x1, y1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

    # show the image
    cv2.imshow('frame', frame)
    print(letters)

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()






