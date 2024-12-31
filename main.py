import cv2
from ultralytics import YOLO
import easyocr

# load the model
yolo = YOLO("predict_model.pt")

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
        # Default color if an unknown class is encountered
        return 128, 128, 128    # Gray


# Initialize counter for saved images
image_counter = 0

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue

    results = yolo.track(frame, stream=True)

    for result in results:
        # get the classes names
        classes_names = result.names

        # iterate over each box
        for box in result.boxes:
            # check if confidence is greater than 40 percent
            if box.conf[0] > 0.4:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # get the class
                cls = int(box.cls[0])

                # get the class name
                class_name = classes_names[cls]

                # get the respective colour
                colour = getColours(cls)

                # draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # put the class name and confidence on the image
                cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

                # Crop the detected object
                cropped_object = frame[y1:y2, x1:x2]

                # Save the cropped image
                if cropped_object.size > 0:  # Check to ensure cropping was successful
                    image_path = f'detected_object_{image_counter}.png'
                    cv2.imwrite(image_path, cropped_object)
                    print(f'Saved: {image_path}')
                    image_counter += 1

    # show the image
    cv2.imshow('frame', frame)

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()






reader = easyocr.Reader(['en'])  # specify the language
result = reader.readtext('image.jpg')

for (bbox, text, prob) in result:
    print(f'Text: {text}, Probability: {prob}')



