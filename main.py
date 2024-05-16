import cv2

thres = 0.52  # Threshold to detect objects


# Open the video capture with camera index 0 (you can change it to 2 if needed)
cap = cv2.VideoCapture(0)

# Set the resolution and frame rate
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# Load class names from names file
classNames = []
classFile = 'names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the pre-trained model  (ssd = single shot multibox detector)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb' #fronzen name is because the weights are fixed and ready for inference(refers to the process of using a trained model to make predictions), meaning the model is no longer being trained or updated.

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)#This line sets the scale factor for the input images. The input image pixels are divided by this value before being passed through the network. This scaling is often used to normalize the input data to a specific range. Here, the scale factor of 1.0 / 127.5 indicates that the pixel values are normalized to the range [0, 2].
net.setInputMean((127.5, 127.5, 127.5))#This line sets the mean values for each channel of the input images.
net.setInputSwapRB(True)

# Set a larger window size
cv2.namedWindow('Output', cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()

    # Check if the image is not empty
    if not success or img is None:
        continue

    # Resize the image to match the expected input size
    img = cv2.resize(img, (320, 320))

    # Detect objects in the image
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Ensure classId is within the valid range
            if 0 <= classId < len(classNames):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)#(0,255,0 is for green color)
                cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), #This line adds text (cv2.putText) to the image indicating the class name of the detected object
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                print(f"Invalid classId: {classId}")

    # Display the image in a larger window
    cv2.imshow('Output', img)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close the window
cap.release()
cv2.destroyAllWindows()

#In the provided code, the activation function used in the object detection model is typically the Rectified Linear Unit (ReLU). However, it's important to note that the choice of activation function might vary depending on the specific layers and configurations within the neural network architecture.

#In general, the ReLU activation function is commonly used in deep learning models, including convolutional neural networks (CNNs) like MobileNet, because of its simplicity and effectiveness. ReLU replaces all negative values in the input with zero, while leaving positive values unchanged.

#MobileNet is a lightweight convolutional neural network architecture designed for mobile and embedded vision applications. It provides a good balance between accuracy and speed, making it suitable for real-time object detection tasks. 

#MobileNet SSD: Think of MobileNet SSD as a smart tool designed to quickly spot different objects in pictures or videos. It's like a combination of a camera and a brain that can recognize things like people, cars, and animals.

#Frozen Inference Graph (frozen_inference_graph.pb): This is like a memory card that stores all the knowledge the smart tool has gained through training. It contains the instructions on how to spot objects and the experiences it has learned from seeing many examples.
