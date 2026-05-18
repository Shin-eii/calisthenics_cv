import cv2
import mediapipe as mp
import numpy as np

# load model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    angle = np.degrees(np.arccos(cosine_angle))

    return angle


cap = cv2.VideoCapture("pushup_3.mov")

# Attibute 
counter = 0
stage = "up"

ELBOW_DOWN = 90
ELBOW_UP = 160
BODY_OK = 155   # relaxed threshold (realistic for CV)
min_body_angle = 180



while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    
    # detect body
    results = pose.process(image_rgb)

    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Set Up 
    elbow_angle = None
    body_angle = None
    body_ok = False

    # if only detect
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Correct Body angle + Elbow angle = --> Count

        # TODO : Find Body Angle

        # Step1: Get Both Side
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

        # Step2: Choose better side
        right_visibility = (rs.visibility + rh.visibility + ra.visibility) / 3
        left_visibility = (ls.visibility + lh.visibility + la.visibility) / 3

        if right_visibility > left_visibility:
            shoulder, hip, ankle = rs, rh, ra
            side_used = "Right"
        else:
            shoulder, hip, ankle = ls, lh, la
            side_used = "Left"

        # Step3: Find body angle
        body_angle = calculate_angle(shoulder, hip, ankle)
        body_ok = body_angle > BODY_OK


        # Test : Put angle degree and color on the pic
        h_img, w_img, _ = image_bgr.shape
        shoulder_xy = (int(shoulder.x * w_img), int(shoulder.y * h_img))
        hip_xy = (int(hip.x * w_img), int(hip.y * h_img))
        ankle_xy = (int(ankle.x * w_img), int(ankle.y * h_img))

        if body_angle > BODY_OK:
            line_color = (0, 255, 0)  # green = good form
            body_status = "Body OK"
        else:
            line_color = (0, 0, 255)  # red = bad form
            body_status = "Fix Body"

        # Draw joints
        cv2.circle(image_bgr, shoulder_xy, 6, line_color, -1)
        cv2.circle(image_bgr, hip_xy, 6, line_color, -1)
        cv2.circle(image_bgr, ankle_xy, 6, line_color, -1)

        # Draw body line (back line)
        cv2.line(image_bgr, shoulder_xy, hip_xy, line_color, 4)
        cv2.line(image_bgr, hip_xy, ankle_xy, line_color, 4)

        cv2.putText(
            image_bgr,
            f"{int(body_angle)} deg",
            (hip_xy[0] + 10, hip_xy[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            line_color,
            2,
        )

        # Get the Elbow Angle
        elbow = None

        # Right arm
        re = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Left arm
        le = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

        # Choose best side
        right_arm_visibility = (rs.visibility + re.visibility + rw.visibility) / 3
        left_arm_visibility  = (ls.visibility + le.visibility + lw.visibility) / 3

        # Select Best Arm
        if right_arm_visibility > left_arm_visibility:
            shoulder, elbow, wrist = rs, re, rw
            arm_used = "Right"
        else:
            shoulder, elbow, wrist = ls, le, lw
            arm_used = "Left"

        # Calculate Elbow Angle
        if shoulder.visibility > 0.8 and elbow.visibility > 0.8 and wrist.visibility > 0.8:
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
        
        h, w, _ = image_bgr.shape
        s_xy = (int(shoulder.x * w), int(shoulder.y * h))
        e_xy = (int(elbow.x * w), int(elbow.y * h))
        w_xy = (int(wrist.x * w), int(wrist.y * h))

        # Draw arm
        cv2.line(image_bgr, s_xy, e_xy, (255, 0, 0), 4)
        cv2.line(image_bgr, e_xy, w_xy, (255, 0, 0), 4)
        cv2.circle(image_bgr, s_xy, 6, (0, 255, 0), -1)
        cv2.circle(image_bgr, e_xy, 6, (0, 255, 0), -1)
        cv2.circle(image_bgr, w_xy, 6, (0, 255, 0), -1)

        # Draw angle
        
            

        

    # DOWN: only enter if not already down
        if elbow_angle is not None:
            if elbow_angle < ELBOW_DOWN and stage == "up":
                stage = "down"

            # UP transition (count rep)
            elif elbow_angle > ELBOW_UP and stage == "down" :
                if body_ok:
                    counter += 1
                stage = "up"

            # draw Elbow Angle
            cv2.putText(
                image_bgr,
                f"{int(elbow_angle)} deg",
                (e_xy[0] + 10, e_xy[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
                        

     # Background box (optional but very clear)
    cv2.rectangle(image_bgr, (10, 10), (220, 120), (0, 0, 0), -1)

    # Stage (up / down)
    cv2.putText(
            image_bgr,
            f"Stage: {stage}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
    )

   

    # Rep count
    cv2.putText(
            image_bgr,
            f"Reps: {counter}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


    cv2.imshow("Calisthenics Pose", image_bgr)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
