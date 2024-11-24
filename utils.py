import cv2
import mediapipe as mp


def get_face_landmarks(image, draw=False, static_image_mode=True):
    """
        Function to extract face landmarks from an image.

        Args:
            image: Input image in BGR format.
            draw: Boolean, whether to draw the landmarks on the image.
            static_image_mode: Boolean, whether the input is a static image (True) or a video frame (False).

        Returns:
            A list of normalized face landmarks relative to the image dimensions.
        """

    # Convert the image from BGR to RGB color space, as Mediapipe expects RGB images.
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize the FaceMesh model with configuration parameters.
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=static_image_mode, # Process static images (True) or video frames (False).
                                                max_num_faces=1, # Detect only one face in the image.
                                                min_detection_confidence=0.2) # Minimum confidence threshold for detection.
    image_rows, image_cols, _ = image.shape # Get the dimensions of the input image.

    # Process the RGB image with the FaceMesh model to detect face landmarks.
    results = face_mesh.process(image_input_rgb)

    # Initialize a list to store the face landmarks.
    image_landmarks = []

    # Check if face landmarks were detected.
    if results.multi_face_landmarks:

        # If draw is True, draw the detected landmarks on the input image.
        if draw:

            mp_drawing = mp.solutions.drawing_utils # Drawing utilities from Mediapipe.
            mp_drawing_styles = mp.solutions.drawing_styles # Predefined drawing styles.
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1) # Style for landmarks.

            # Draw landmarks and connections on the input image.
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0], # Landmarks of the first detected face.
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS, # Face contour connections.
                landmark_drawing_spec=drawing_spec, # Style for individual landmarks.
                connection_drawing_spec=drawing_spec) # Style for landmark connections.

        # Extract the landmarks of the first detected face.
        ls_single_face = results.multi_face_landmarks[0].landmark

        # Initialize separate lists to store x, y, and z coordinates of landmarks.
        xs_ = []
        ys_ = []
        zs_ = []

        # Loop through each landmark and collect its coordinates.
        for idx in ls_single_face:
            xs_.append(idx.x)
            ys_.append(idx.y)
            zs_.append(idx.z)

        # Normalize the coordinates by subtracting the minimum value in each dimension.
        for j in range(len(xs_)):
            image_landmarks.append(xs_[j] - min(xs_))
            image_landmarks.append(ys_[j] - min(ys_))
            image_landmarks.append(zs_[j] - min(zs_))

    return image_landmarks