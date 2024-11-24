import os

import cv2
import numpy as np

from utils import get_face_landmarks


data_dir = './data'

output = []
# Loop through each emotion directory, sorted alphabetically for consistent labeling.
for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    # Loop through each image in the current emotion directory.
    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
        # Construct the full path to the current image file.
        image_path = os.path.join(data_dir, emotion, image_path_)

        # Read the image using OpenCV.
        image = cv2.imread(image_path)

        # Extract face landmarks from the image using the utility function.
        face_landmarks = get_face_landmarks(image)

        # Check if the detected face landmarks match the expected size (468 points × 3 coordinates).
        if len(face_landmarks) == 1404:
            # Append the emotion index (as an integer) to the face landmarks.
            face_landmarks.append(int(emotion_indx))
            # Add the face landmarks with the emotion index to the output list.
            output.append(face_landmarks)

# Save the processed data (landmarks and labels) to a text file using NumPy.
np.savetxt('data.txt', np.asarray(output))
