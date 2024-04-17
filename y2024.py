from tkinter import messagebox
from ultralytics import YOLO
import cv2
import math

def count_people():
    # start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # model
    model = YOLO("yolo-Weights/yolov8n.pt")

    # object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
    # Counter for people
    people_count = 0



    while True:
        success, img = cap.read()
        if not success:
            break  # If the frame is not successfully read, break the loop

        results = model(img, stream=True)

        # Process coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integer values

                # Calculate confidence and check if it's greater than 50%
                confidence = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])
                if confidence > 0.5 and cls == 0:  # Check if confidence is greater than 0.5
                    people_count += 1  # Increment the people count


                    # Draw bounding box on the image
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # Display object details on the image
                    org = (x1, y1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.putText(img, f"{classNames[cls]}: {confidence:.2f}", org, font, fontScale, color, thickness)
        cv2.putText(img, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return people_count

# Call the function and get the number of people detected
detected_people = count_people()
# Using tkinter to display a message box with the number of people detected
import tkinter as tk
root = tk.Tk()
root.withdraw()  # Hide the main window
messagebox.showinfo("People Count", f"Number of People Detected: {detected_people}")
root.destroy()