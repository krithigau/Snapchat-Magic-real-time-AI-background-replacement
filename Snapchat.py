import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Load background image (replace 'background.jpg' with your image path)
background_image = cv2.imread('space.png')
if background_image is None:
    raise FileNotFoundError("Make sure 'background image' exists in the same folder.")

# Open webcam
cap = cv2.VideoCapture(0)

# Buffer for temporal smoothing
N = 5
mask_buffer = []

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(rgb_frame)

    mask = results.segmentation_mask

    # Add mask to buffer and average the masks
    mask_buffer.append(mask)
    if len(mask_buffer) > N:
        mask_buffer.pop(0)

    avg_mask = np.mean(mask_buffer, axis=0)
    condition = np.stack((avg_mask,) * 3, axis=-1) > 0.5

    # Resize background image to match frame size
    bg_resized = cv2.resize(background_image, (width, height))

    # Composite the final image
    output_frame = np.where(condition, frame, bg_resized)

    # Smooth the edges of the mask
    blur_mask = cv2.GaussianBlur((avg_mask * 255).astype(np.uint8), (25, 25), 0)
    blur_mask = cv2.cvtColor(blur_mask, cv2.COLOR_GRAY2BGR) / 255.0
    output_frame = (output_frame * blur_mask + bg_resized * (1 - blur_mask)).astype(np.uint8)

    # Display the result
    cv2.imshow('Virtual Background (Smooth)', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup and release resources
cap.release()
cv2.destroyAllWindows()
