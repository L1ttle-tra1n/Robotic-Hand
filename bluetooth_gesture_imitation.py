# this file is for gesture imitation and transfer data to esp32 via bluetooth

import cv2
import mediapipe as mp
import math
import asyncio
from bleak import BleakClient

ESP32_BLE_ADDRESS = "9B07F07D-313C-AA11-01F1-4C6F31D19CDC"  # alter if u have new address
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    dot_product = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2]
    length_ba = math.sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2)
    length_bc = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)
    if length_ba * length_bc == 0:
        return 0.0
    cos_angle = dot_product / (length_ba * length_bc)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle = math.degrees(math.acos(cos_angle))
    return angle

finger_joint_triplets = {
    'Thumb':  [(0, 1, 2),  (1, 2, 3),  (2, 3, 4)],
    'Index':  [(0, 5, 6),  (5, 6, 7),  (6, 7, 8)],
    'Middle': [(0, 9, 10), (9, 10, 11), (10, 11, 12)],
    'Ring':   [(0, 13, 14),(13, 14, 15),(14, 15, 16)],
    'Pinky':  [(0, 17, 18),(17, 18, 19),(18, 19, 20)]
}

def calculate_wrist_rotation(landmarks, image_width, image_height, hand_label):
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    index_x, index_y = int(index_mcp[0] * image_width), int(index_mcp[1] * image_height)
    pinky_x, pinky_y = int(pinky_mcp[0] * image_width), int(pinky_mcp[1] * image_height)
    dx = pinky_x - index_x
    dy = pinky_y - index_y
    palm_angle = math.degrees(math.atan2(dy, dx))
    if hand_label == "Right":
        wrist_rotation = -palm_angle
    else:
        wrist_rotation = palm_angle
    if wrist_rotation > 180:
        wrist_rotation -= 360
    elif wrist_rotation < -180:
        wrist_rotation += 360
    wrist = landmarks[0]
    wrist_coords = (int(wrist[0] * image_width), int(wrist[1] * image_height))
    return wrist_rotation, wrist_coords

def calculate_wrist_pitch(landmarks):
    p0 = landmarks[0]
    p5 = landmarks[5]
    p17 = landmarks[17]

    v1 = (p5[0] - p0[0], p5[1] - p0[1], p5[2] - p0[2])
    v2 = (p17[0] - p0[0], p17[1] - p0[1], p17[2] - p0[2])

    n_x = v1[1]*v2[2] - v1[2]*v2[1]
    n_y = v1[2]*v2[0] - v1[0]*v2[2]
    n_z = v1[0]*v2[1] - v1[1]*v2[0]

    ref = (0.0, 0.0, 1.0)
    dot = n_x*ref[0] + n_y*ref[1] + n_z*ref[2]

    if dot < 0:
        n_x, n_y, n_z = -n_x, -n_y, -n_z

    measured_pitch = math.degrees(math.atan2(n_y, n_z))
    measured_pitch = -measured_pitch

    max_compensation = 20.0
    threshold = 50.0
    compensation = (abs(measured_pitch) / threshold) * max_compensation
    if compensation > max_compensation:
        compensation = max_compensation

    if measured_pitch >= 0:
        final_pitch = measured_pitch + compensation
    else:
        final_pitch = measured_pitch - compensation

    return final_pitch



async def main():
    print("Connecting")
    client = BleakClient(ESP32_BLE_ADDRESS)
    await client.connect()
    print("Connected!")

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            h, w, _ = frame.shape

            send_data_str = ""

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )

                    handedness = results.multi_handedness[hand_idx].classification[0]
                    hand_label = handedness.label
                    hand_score = handedness.score

                    wrist_x = int(hand_landmarks.landmark[0].x * w)
                    wrist_y = int(hand_landmarks.landmark[0].y * h)
                    cv2.putText(frame, f"{hand_label} {hand_score:.2f}",
                                (wrist_x - 50, wrist_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                    finger_angles = []
                    for finger_name, triplets in finger_joint_triplets.items():
                        for (a, b, c) in triplets:
                            angle = calculate_angle(landmarks[a], landmarks[b], landmarks[c])
                            finger_angles.append(angle)
                            bx = int(landmarks[b][0] * w)
                            by = int(landmarks[b][1] * h)
                            cv2.putText(frame, f"{int(angle)}", (bx, by),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    wrist_rotation, wrist_coords = calculate_wrist_rotation(landmarks, w, h, hand_label)
                    pitch = calculate_wrist_pitch(landmarks)
                    wx, wy = wrist_coords
                    cv2.putText(frame, f"Wrist Yaw: {int(wrist_rotation)} deg",
                                (wx, wy - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Wrist Pitch: {int(pitch)} deg",
                                (wx, wy - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    send_data_str = f"{hand_label} "  \
                                    f"Yaw={int(wrist_rotation)} " \
                                    f"Pitch={int(pitch)} " \
                                    f"Fingers={','.join(str(int(a)) for a in finger_angles)}\n"

            if send_data_str:
                try:
                    await client.write_gatt_char(CHARACTERISTIC_UUID, send_data_str.encode())
                except Exception as e:
                    print("BLE Write Error:", e)

            cv2.imshow("Hand Tracking with BLE", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            await asyncio.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

    if client.is_connected:
        await client.disconnect()
    print("Disconnected")

if __name__ == "__main__":
    asyncio.run(main())
