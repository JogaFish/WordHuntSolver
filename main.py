import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

# load yolo model
yolo = YOLO("predict_model.pt")

# load OCR reader model
reader = easyocr.Reader(['en'])

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


image_counter = 0

while True:
    ret, frame = videoCap.read()
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
            if box.conf[0] > 0.70:
                # get coordinates and convert to integers
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class, name and colour
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                colour = getColours(cls)

                if cls == 1:
                    # Crop the detected object
                    cropped_object = frame[y1:y2, x1:x2]

                    # Save the cropped image
                    if cropped_object.size > 0:                 # Check to ensure cropping was successful
                        img_path = f'temp_img_{image_counter}.png'
                        cv2.imwrite(img_path, cropped_object)
                        ocr_result = reader.readtext(img_path, low_text=0.3, contrast_ths=0.5,
                                                     allowlist=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                                                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                                                                'U', 'V', 'W', 'X', 'Y', 'Z'])
                        image_counter += 1
                        if ocr_result:
                            conf = ocr_result[0][2]
                            letter = ocr_result[0][1]
                            letters += letter

                            # draw the rectangle
                            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                            # put the class name and confidence on the image
                            # cv2.putText(frame, f'{letter} {conf:.2f}', (x1, y1),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                        else:
                            colour = getColours(-1)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                            continue

    # show the image
    cv2.imshow('frame', frame)
    print(letters)

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()






