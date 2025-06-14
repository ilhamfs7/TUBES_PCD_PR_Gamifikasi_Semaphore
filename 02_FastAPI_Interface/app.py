import cv2
import uvicorn
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
import semaphore as game
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()

# Izinkan akses dari semua asal (penting untuk Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi webcam dengan pengecekan
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Cannot open webcam")
    raise RuntimeError("Cannot open webcam")

# Load data pemain saat startup
game.load_player_stats()

def process_frame_generator():
    """Generator yang memproses frame, menjalankan logika game, dan menghasilkan frame untuk streaming."""
    # Inisialisasi model MediaPipe
    pose_model = mp.solutions.pose.Pose(model_complexity=0)  # Model ringan untuk performa
    hands_model = mp.solutions.hands.Hands(max_num_hands=2)

    # Gunakan last_frames dari semaphore.py untuk konsistensi
    global game

    # Konfigurasi dari game_logic
    FLIP = True  # Selfie view, konsisten dengan --flip
    DRAW_LANDMARKS = True  # Gambar landmarks, konsisten dengan --landmarks
    DISPLAY_ONLY = False  # Ketik output, konsisten dengan --type
    ALLOW_REPEAT = True  

    # Variabel untuk indikator benar/salah
    feedback_message = ""
    feedback_timer = 0
    FEEDBACK_DURATION = game.FPS * 2  # Tampilkan feedback selama 2 detik

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                logging.warning("Gagal membaca frame dari webcam, menghentikan stream.")
                break

            # Flip gambar untuk tampilan selfie
            if FLIP:
                image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Proses dengan MediaPipe
            pose_results = pose_model.process(image_rgb)
            hand_results = hands_model.process(image_rgb)

            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Gambar landmarks
            if DRAW_LANDMARKS and pose_results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                    game.DEFAULT_LANDMARKS_STYLE)
            if DRAW_LANDMARKS and hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                        game.DEFAULT_LANDMARKS_STYLE, game.DEFAULT_HAND_CONNECTIONS_STYLE)

            # Cool-off period: skip jika ada frame yang baru saja ditandatangani
            if any(frame['signed'] for frame in game.last_frames):
                pass
            else:
                # Proses tangan
                hands = []
                hand_index = 0
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        hands.append([])
                        for point in hand_landmarks.landmark:
                            hands[hand_index].append({'x': 1 - point.x, 'y': 1 - point.y})  # Konsisten dengan semaphore.py
                        hand_index += 1

                if pose_results.pose_landmarks:
                    # Update last_frames
                    game.last_frames = game.last_frames[1:] + [game.empty_frame.copy()]

                    body = []
                    for point in pose_results.pose_landmarks.landmark:
                        body.append({'x': 1 - point.x, 'y': 1 - point.y, 'visibility': point.visibility})  # Konsisten dengan semaphore.py

                    # Deteksi backspace
                    mouth = (body[9], body[10])
                    palms = (body[19], body[20])
                    if game.is_mouth_covered(mouth, palms):
                        game.output(['backspace'], image, DISPLAY_ONLY)
                        if len(game.typed_word) > 0:
                            game.typed_word = game.typed_word[:-1]

                    # Deteksi command dan control
                    legL = (body[23], body[25], body[27])
                    command_on = game.is_leg_lifted(legL)
                    legR = (body[24], body[26], body[28])
                    control_on = game.is_leg_lifted(legR)

                    # Definisikan lengan
                    shoulderL, elbowL, wristL = body[11], body[13], body[15]
                    armL = (shoulderL, elbowL, wristL)
                    shoulderR, elbowR, wristR = body[12], body[14], body[16]
                    armR = (shoulderR, elbowR, wristR)

                    mouth_width = abs(mouth[1]['x'] - mouth[0]['x'])

                    # Deteksi arrow keys
                    if game.is_arms_crossed(elbowL, wristL, elbowR, wristR, mouth_width):
                        if game.is_limb_pointing(*legL) and game.is_limb_pointing(*legR):
                            legL_angle = game.get_limb_direction(legL, game.LEG_ARROW_ANGLE)
                            legR_angle = game.get_limb_direction(legR, game.LEG_ARROW_ANGLE)
                            leg_arrow = game.leg_arrow_angles.get((legL_angle, legR_angle), '')
                            if leg_arrow:
                                game.output([leg_arrow + ' arrow'], image, DISPLAY_ONLY)

                    # Deteksi shift
                    shift_on = False
                    min_finger_reach = game.FINGER_MOUTH_RATIO * mouth_width
                    palmL, palmR = body[17], body[18]
                    for hand in hands:
                        thumb, forefinger, pinky = hand[4], hand[8], hand[20]
                        hand_open = game.is_hand_open(thumb, forefinger, pinky, palmL, palmR, min_finger_reach)
                        shift_on = shift_on or hand_open  # Perbaikan logika shift

                    # Deteksi numerals (squat)
                    kneeL, kneeR = body[25], body[26]
                    hipL, hipR = body[23], body[24]
                    numerals = game.is_squatting(hipL, kneeL, hipR, kneeR)

                    # Deteksi alphanumeric
                    if game.is_limb_pointing(*armL) and game.is_limb_pointing(*armR):
                        armL_angle = game.get_limb_direction(armL)
                        armR_angle = game.get_limb_direction(armR)
                        if game.type_semaphore(armL_angle, armR_angle, image, shift_on, numerals, 
                                              command_on, control_on, DISPLAY_ONLY, ALLOW_REPEAT):
                            game.last_frames[-1]['signed'] = True
                            # Cek apakah kata selesai (space diketik)
                            if game.current_semaphore == "space":
                                if game.typed_word == game.target_word:
                                    feedback_message = "✅ Benar!"
                                    logging.info(f"Correct word: {game.typed_word}")
                                else:
                                    feedback_message = "❌ Salah!"
                                    logging.info(f"Incorrect word: {game.typed_word}, expected: {game.target_word}")
                                feedback_timer = FEEDBACK_DURATION

                    # Deteksi jump
                    if game.is_jumping(hipL, hipR):
                        game.output(game.last_keys, image, DISPLAY_ONLY)

            # Update feedback timer
            if feedback_timer > 0:
                feedback_timer -= 1
                cv2.putText(image, feedback_message, (image.shape[1]//2 - 100, image.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if "Benar" in feedback_message else (0, 0, 255), 3)

            # Gambar UI
            game.display_stats(image)
            game.display_word_challenge(image)

            # Encode frame ke JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", image)
            if not flag:
                logging.warning("Failed to encode frame to JPEG")
                continue

            # Yield frame dalam format byte untuk streaming
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) + b'\r\n')

    except Exception as e:
        logging.error(f"Error in process_frame_generator: {e}")
    finally:
        pose_model.close()
        hands_model.close()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(process_frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/game_state")
def game_state():
    return {
        "player_stats": game.player_stats,
        "target_word": game.target_word,
        "typed_word": game.typed_word,
        "game_mode": game.game_mode,
        "current_difficulty": game.current_difficulty,
        "challenge_index": game.challenge_index,
        "challenge_words": game.challenge_words
    }

@app.post("/start_game/{mode}")
def start_game(mode: str):
    valid_modes = ["practice", "challenge"]
    if mode not in valid_modes:
        logging.warning(f"Invalid mode attempted: {mode}")
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}. Valid modes: {valid_modes}")
    game.start_game_mode(mode, game.current_difficulty)
    logging.info(f"Started {mode} mode with difficulty {game.current_difficulty}")
    return {"message": f"{mode} mode started with difficulty {game.current_difficulty}"}

@app.post("/set_difficulty/{difficulty}")
def set_difficulty(difficulty: str):
    if difficulty not in game.WORD_LISTS:
        logging.warning(f"Invalid difficulty attempted: {difficulty}")
        raise HTTPException(status_code=400, detail=f"Invalid difficulty: {difficulty}. Valid difficulties: {list(game.WORD_LISTS.keys())}")
    game.current_difficulty = difficulty
    if game.game_mode in ["practice", "challenge"]:
        game.start_game_mode(game.game_mode, difficulty)
    logging.info(f"Difficulty set to {difficulty}")
    return {"message": f"Difficulty set to {difficulty}"}

@app.on_event("shutdown")
def cleanup():
    """Pembersihan sumber daya saat server dimatikan."""
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Server shutdown, resources released")

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logging.error(f"Error running server: {e}")
        cap.release()
        cv2.destroyAllWindows()