import cv2

thres = 0.45  # Threshold to detect objects

# Load class names from names file
classNames = []
classFile = 'names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the pre-trained model (ssd = single shot multibox detector)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Function to detect objects in an image
def detect_objects(image):
    # Detect objects in the image
    classIds, confs, bbox = net.detect(image, confThreshold=thres)

    if len(classIds) != 0:
        object_counts = {}
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Ensure classId is within the valid range
            if 0 <= classId < len(classNames):
                class_name = classNames[classId - 1].upper()
                cv2.rectangle(image, box, color=(0, 255, 0), thickness=2)
                cv2.putText(image, class_name, (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                # Count the objects of each class
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
            else:
                print(f"Invalid classId: {classId}")

        # Check for specific conditions based on object counts
        if 'TV' in object_counts and object_counts['TV'] > 1:
            cv2.putText(image, 'Computer Lab', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        elif 'person' in object_counts and object_counts['person'] > 1:
            cv2.putText(image, 'football', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return image

# Function to process the video stream
def process_video():
    # Open the video capture with camera index 0
    cap = cv2.VideoCapture(0)
    # Set the resolution and frame rate
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(10, 70)

    # Define the scale factor for resizing the output window
    scale_factor = 1.5

    while True:
        success, img = cap.read()

        # Check if the image is not empty
        if not success or img is None:
            continue

        # Resize the image to match the expected input size
        img = cv2.resize(img, (320, 320))

        # Detect objects in the image
        img = detect_objects(img)

        # Resize the output window
        window_height, window_width = img.shape[:2]
        cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Output', int(window_width * scale_factor), int(window_height * scale_factor))

        # Display the image in a larger window
        cv2.imshow('Output', img)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

# Function to process a single image
def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image is not empty
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Resize the image to match the expected input size
    image = cv2.resize(image, (320, 320))

    # Detect objects in the image
    image = detect_objects(image)

    # Display the image
    cv2.imshow('Output', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    print("Choose an option:")
    print("1. Process video stream")
    print("2. Process a single image")
    option = input("Enter option number: ")

    if option == '1':
        process_video()
    elif option == '2':
        image_path = input("Enter path to the image: ")
        process_image(image_path)
    else:
        print("Invalid option.")
