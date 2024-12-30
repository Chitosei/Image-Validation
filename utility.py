import cv2
import numpy as np


def extract_eye_coordinates(landmarks, indices):
    return [landmarks[i] for i in indices]


def calculate_ear(landmarks):
    # Define critical indices for each eye
    left_eye_vertical_top = 386
    left_eye_vertical_bottom = 374
    left_eye_horizontal_left = 263
    left_eye_horizontal_right = 362

    top_left = np.array(landmarks[left_eye_vertical_top])
    bottom_left = np.array(landmarks[left_eye_vertical_bottom])
    left_left = np.array(landmarks[left_eye_horizontal_left])
    right_left = np.array(landmarks[left_eye_horizontal_right])

    # Calculate vertical and horizontal distances
    vertical_left = np.linalg.norm(top_left - bottom_left)
    horizontal_left = np.linalg.norm(left_left - right_left)

    result_left = vertical_left / horizontal_left
    # Right

    right_eye_vertical_top = 159
    right_eye_vertical_bottom = 145
    right_eye_horizontal_left = 33
    right_eye_horizontal_right = 133

    top_right = np.array(landmarks[right_eye_vertical_top])
    bottom_right = np.array(landmarks[right_eye_vertical_bottom])
    left_right = np.array(landmarks[right_eye_horizontal_left])
    right_right = np.array(landmarks[right_eye_horizontal_right])

    # Calculate vertical and horizontal distances
    vertical_right = np.linalg.norm(top_right - bottom_right)
    horizontal_right = np.linalg.norm(left_right - right_right)

    result_right = vertical_right / horizontal_right

    ear = [result_left, result_right]
    return ear


def visualize_landmarks(image, face_landmarks, draw_indices=None):
    """
    Visualizes facial landmarks on the image.

    Parameters:
        image (numpy.ndarray): The original image.
        face_landmarks: Mediapipe facial landmarks result.
        draw_indices (list): Specific indices to highlight (optional).
    """
    # Convert image to BGR (for OpenCV visualization)
    annotated_image = np.array(image.copy())

    # Get image dimensions
    height, width, _ = annotated_image.shape

    # Loop through all landmarks
    for idx, landmark in enumerate(face_landmarks.landmark):
        # Convert normalized coordinates to pixel values
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        # Draw all landmarks or specific ones if draw_indices is set
        if draw_indices is None or idx in draw_indices:
            cv2.circle(annotated_image, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

    return annotated_image


def head_pose_estimation(img_array, landmarks):
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eyes corner
        (225.0, 170.0, -135.0),  # Right eyes corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ], dtype="double")

    # Extract corresponding 2D image points from MediaPipe landmarks
    image_points = np.array([
        landmarks[1],  # Nose
        landmarks[152],  # Chin
        landmarks[33],  # Left eyes corner
        landmarks[263],  # Right eyes corner
        landmarks[61],  # Left mouth corner
        landmarks[291]  # Right mouth corner
    ], dtype="double")

    # Distortion coefficients
    dist_coeffs = np.zeros((4, 1))

    # Camera matrix
    focal_length = max(img_array.shape[1], img_array.shape[0])
    center = (img_array.shape[1] / 2, img_array.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs
    )
    # Calculate the yaw, pitch and roll angle
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
    yaw, pitch, roll = euler_angles.flatten()
    # Clarity
    yaw_true_value = yaw + 180
    print([yaw, pitch, roll])
    print([yaw_true_value, pitch, roll])

    return [yaw_true_value, pitch, roll]

# # Example usage:
# # Assuming `results` is the output of `face_mesh.process(image)`
# if results.multi_face_landmarks:
#     for face_landmarks in results.multi_face_landmarks:
#         # Visualize all landmarks
#         annotated_image = visualize_landmarks(image, face_landmarks)
#
#         # Optionally save or display the annotated image
#         cv2.imshow('Landmarks', annotated_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
