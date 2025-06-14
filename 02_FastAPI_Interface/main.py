from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
import json
import time
import base64
from io import BytesIO

# Import semua fungsi dan konstanta dari file semaphore.py Anda
# Pastikan semua variabel global yang diperlukan diinisialisasi di sini atau dilewatkan sebagai parameter
# atau disimpan dalam dictionary state per sesi.
from semaphore import (
    get_angle, is_missing, is_limb_pointing, get_limb_direction,
    is_arm_crossed, is_arms_crossed, is_leg_lifted, is_jumping,
    is_mouth_covered, is_squatting, is_finger_out, is_hand_open,
    type_semaphore, get_key_text, load_player_stats, save_player_stats,
    get_random_word, calculate_accuracy, add_experience, check_achievement,
    DEFAULT_LANDMARKS_STYLE, DEFAULT_HAND_CONNECTIONS_STYLE, SEMAPHORES,
    LEG_ARROW_ANGLE, FRAME_HISTORY, HALF_HISTORY, QUARTER_HISTORY,
    VISIBILITY_THRESHOLD, STRAIGHT_LIMB_MARGIN, EXTENDED_LIMB_MARGIN,
    LEG_LIFT_MIN, ARM_CROSSED_RATIO, MOUTH_COVER_THRESHOLD, SQUAT_THRESHOLD,
    JUMP_THRESHOLD, FINGER_MOUTH_RATIO, WORD_LISTS, LEVEL_REQUIREMENTS,
    EXPERIENCE_PER_WORD, STREAK_BONUS, ACCURACY_BONUS, ACHIEVEMENTS
)

app = FastAPI()

# Inisialisasi model MediaPipe di luar loop pemrosesan untuk efisiensi
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_model = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Manajemen state per sesi pengguna
# Untuk kesederhanaan, kita bisa menggunakan dictionary global, tapi dalam aplikasi nyata
# Anda akan menggunakan database atau sistem manajemen sesi yang lebih canggih.
session_states = {}

class GameState:
    def __init__(self, session_id):
        self.session_id = session_id
        self.player_stats = {
            "level": 1,
            "experience": 0,
            "streak": 0,
            "max_streak": 0,
            "total_words": 0,
            "accuracy": 100.0,
            "achievements": [],
            "coins": 0
        }
        self.current_difficulty = "mudah"
        self.start_time = None
        self.game_mode = "practice"
        self.challenge_words = []
        self.challenge_index = 0
        self.time_limit = 60
        self.words_completed_in_session = 0
        self.target_word = ""
        self.typed_word = ""
        self.current_semaphore = ""
        self.last_keys = []
        self.last_frames = FRAME_HISTORY * [{
            'hipL_y': 0, 'hipR_y': 0, 'hips_dy': 0,
            'dxL_thrust_hipL': 0, 'dxL_thrust_hipR': 0,
            'dxR_thrust_hipL': 0, 'dxR_thrust_hipR': 0,
            'signed': False,
        }]
        self.frame_midpoint = (0, 0) # Ini akan diatur oleh frontend

        # Muat statistik pemain dari file saat inisialisasi
        # Anda perlu memodifikasi load_player_stats dan save_player_stats
        # agar menerima session_id atau path file yang unik.
        # Untuk demo, kita abaikan persistensi file per sesi untuk saat ini
        # load_player_stats() # Ini mungkin perlu dimodifikasi

    # Metode untuk update player stats (sama seperti fungsi di file asli, tapi mengacu ke self.player_stats)
    def add_experience_session(self, amount):
        current_level = self.player_stats["level"]
        self.player_stats["experience"] += amount
        while (current_level < len(LEVEL_REQUIREMENTS) and
               self.player_stats["experience"] >= LEVEL_REQUIREMENTS[current_level - 1]): # Adjust index
            current_level += 1
            self.player_stats["level"] = current_level
            self.player_stats["coins"] += 100 * current_level
            print(f"[{self.session_id}] 🎉 LEVEL UP! Sekarang level {current_level}!")
            self.check_achievement_session(f"level_{current_level}")

    def check_achievement_session(self, achievement_id):
        if achievement_id in ACHIEVEMENTS and achievement_id not in self.player_stats["achievements"]:
            achievement = ACHIEVEMENTS[achievement_id]
            self.player_stats["achievements"].append(achievement_id)
            self.player_stats["coins"] += achievement["reward"]
            print(f"[{self.session_id}] 🏆 ACHIEVEMENT UNLOCKED: {achievement['name']}")
            print(f"[{self.session_id}]   {achievement['desc']} (+{achievement['reward']} coins)")

    def calculate_accuracy_session(self):
        if self.player_stats["total_words"] == 0:
            return 100.0
        return (self.player_stats["streak"] / self.player_stats["total_words"]) * 100

    def get_random_word_session(self, difficulty="mudah"):
        return random.choice(WORDS_LISTS[difficulty]) # Menggunakan WORDS_LISTS karena WORD_LISTS sudah ada di file utama
    
    def generate_challenge_session(self):
        num_words = 5 if self.current_difficulty == "mudah" else 8 if self.current_difficulty == "sedang" else 10
        self.challenge_words = [self.get_random_word_session(self.current_difficulty) for _ in range(num_words)]
        self.challenge_index = 0
        print(f"[{self.session_id}] 🎯 Challenge Mode: {num_words} kata tingkat {self.current_difficulty}")
        print(f"[{self.session_id}] Kata-kata: {', '.join(self.challenge_words)}")


    def type_and_remember_session(self, image, shift_on, numerals, command_on, control_on, allow_repeat):
        if len(self.current_semaphore) == 0 or not self.target_word:
            return

        keys = []
        if shift_on:
            keys.append('shift')
        if command_on:
            keys.append('command')
        if control_on:
            keys.append('control')

        keys.append(self.current_semaphore)

        if allow_repeat or (keys != self.last_keys):
            self.last_keys = keys.copy()
            if self.current_semaphore == "space":
                word_time = time.time() - self.start_time if self.start_time else 0
                
                print(f"[{self.session_id}] Kata: {self.typed_word}")
                if self.typed_word == self.target_word:
                    print(f"[{self.session_id}] ✅ Benar!")
                    self.player_stats["total_words"] += 1
                    self.player_stats["streak"] += 1
                    self.words_completed_in_session += 1
                    
                    if self.player_stats["streak"] > self.player_stats["max_streak"]:
                        self.player_stats["max_streak"] = self.player_stats["streak"]
                    
                    base_exp = EXPERIENCE_PER_WORD
                    streak_bonus = min(self.player_stats["streak"] * STREAK_BONUS, 200)
                    speed_bonus = max(0, 50 - int(word_time)) if word_time > 0 else 0
                    total_exp = base_exp + streak_bonus + speed_bonus
                    
                    self.add_experience_session(total_exp)
                    self.player_stats["coins"] += 10 + (self.player_stats["streak"] // 5)
                    
                    print(f"[{self.session_id}] 💰 +{total_exp} XP, +{10 + (self.player_stats['streak'] // 5)} coins")
                    print(f"[{self.session_id}] ⚡ Streak: {self.player_stats['streak']} | Waktu: {word_time:.1f}s")
                    
                    if self.player_stats["total_words"] == 1:
                        self.check_achievement_session("first_word")
                    if self.player_stats["streak"] == 5:
                        self.check_achievement_session("streak_5")
                    elif self.player_stats["streak"] == 10:
                        self.check_achievement_session("streak_10")
                    if word_time <= 10 and word_time > 0:
                        self.check_achievement_session("speed_demon")
                    
                    self.player_stats["accuracy"] = self.calculate_accuracy_session()
                    if self.player_stats["accuracy"] >= 95:
                        self.check_achievement_session("accuracy_95")
                    if self.player_stats["accuracy"] >= 100:
                        self.check_achievement_session("accuracy_100")
                    
                    if self.game_mode == "challenge":
                        self.challenge_index += 1
                        if self.challenge_index < len(self.challenge_words):
                            self.target_word = self.challenge_words[self.challenge_index]
                            print(f"[{self.session_id}] 🎯 Kata berikutnya: {self.target_word}")
                        else:
                            print(f"[{self.session_id}] 🎉 CHALLENGE SELESAI!")
                            self.add_experience_session(100)
                            self.game_mode = "practice"
                            self.target_word = self.get_random_word_session(self.current_difficulty)
                    else:
                        self.target_word = self.get_random_word_session(self.current_difficulty)
                        print(f"[{self.session_id}] 🎯 Kata baru: {self.target_word}")
                            
                else:
                    print(f"[{self.session_id}] ❌ Salah! Coba lagi.")
                    self.player_stats["streak"] = 0
                    self.player_stats["total_words"] += 1
                    self.player_stats["accuracy"] = self.calculate_accuracy_session()
                    print(f"[{self.session_id}] 🎯 Kata tetap: {self.target_word}")
                    
                self.typed_word = ""
                self.start_time = time.time()
                # save_player_stats() # Ini perlu disesuaikan untuk setiap sesi
                
            else:
                self.typed_word += self.current_semaphore
                if not self.start_time:
                    self.start_time = time.time()
                
            self.current_semaphore = ''
        
        # Output visual (akan digambar di frontend)
        # self.output(keys, image, True) # Ini akan menjadi bagian dari JSON response


@app.post("/process_frame/{session_id}")
async def process_frame(session_id: str, file: UploadFile = File(...)):
    if session_id not in session_states:
        session_states[session_id] = GameState(session_id)
        print(f"New session created: {session_id}")
    
    state = session_states[session_id]

    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Mirror the image horizontally to make it like a selfie camera
    image = cv2.flip(image, 1) # Flip first for consistent landmark prediction relative to display

    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose_model.process(image_rgb)
    hand_results = hands_model.process(image_rgb)
    image.flags.writeable = True
    
    # Dapatkan dimensi frame untuk frame_midpoint
    height, width, _ = image.shape
    state.frame_midpoint = (int(width/2), int(height/2))

    # Draw landmarks (akan digambar di frontend)
    # mp.solutions.drawing_utils.draw_landmarks(...)

    hands = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # draw hands (akan digambar di frontend)
            # mp.solutions.drawing_utils.draw_landmarks(...)
            hand_data = []
            for point in hand_landmarks.landmark:
                hand_data.append({
                    'x': 1 - point.x, # Flip x-coordinate for selfie view consistency
                    'y': 1 - point.y
                })
            hands.append(hand_data)

    landmarks_data = []
    if pose_results.pose_landmarks:
        # Update last_frames and skip if a sign was just made (cool down)
        state.last_frames = state.last_frames[1:] + [{
            'hipL_y': 0, 'hipR_y': 0, 'hips_dy': 0,
            'dxL_thrust_hipL': 0, 'dxL_thrust_hipR': 0,
            'dxR_thrust_hipL': 0, 'dxR_thrust_hipR': 0,
            'signed': False,
        }]

        if any(frame['signed'] for frame in state.last_frames):
            # Jika ada tanda yang baru saja dibuat, lewati pemrosesan gesture lain
            # dan hanya kirim kembali state saat ini.
            # Ini adalah bagian yang tricky karena feedback visual harus tetap ada.
            pass # Lanjutkan pemrosesan untuk menggambar, tapi jangan proses gesture baru
        else:
            for point in pose_results.pose_landmarks.landmark:
                landmarks_data.append({
                    'x': 1 - point.x, # Flip x-coordinate for selfie view consistency
                    'y': 1 - point.y,
                    'visibility': point.visibility
                })

            body = landmarks_data

            # cover mouth: backspace
            mouth = (body[9], body[10])
            palms = (body[19], body[20])
            if is_mouth_covered(mouth, palms):
                # Ini harus memicu penghapusan karakter di typed_word
                if state.typed_word:
                    state.typed_word = state.typed_word[:-1]
                state.last_frames[-1]['signed'] = True
                state.current_semaphore = "backspace" # Set ini agar display tahu apa yang terjadi

            # command: left leg lift
            legL = (body[23], body[25], body[27]) # L hip, knee, ankle
            command_on = is_leg_lifted(legL)

            # control: right leg lift
            legR = (body[24], body[26], body[28]) # R hip, knee, ankle
            control_on = is_leg_lifted(legR)

            shoulderL, elbowL, wristL = body[11], body[13], body[15]
            armL = (shoulderL, elbowL, wristL)

            shoulderR, elbowR, wristR = body[12], body[14], body[16]
            armR = (shoulderR, elbowR, wristR)

            mouth_width = abs(mouth[1]['x']-mouth[0]['x'])

            # arrow keys: arms crossed + leg angles
            # Implementasi ini tidak langsung mengetik, hanya mengatur current_semaphore
            if is_arms_crossed(elbowL, wristL, elbowR, wristR, mouth_width):
                if is_limb_pointing(*legL) and is_limb_pointing(*legR):
                    legL_angle = get_limb_direction(legL, LEG_ARROW_ANGLE)
                    legR_angle = get_limb_direction(legR, LEG_ARROW_ANGLE)
                    leg_arrow_angles = { # Definisikan di sini atau impor
                        (-90, -90 + LEG_ARROW_ANGLE): "right",
                        (-90, -90 + 2*LEG_ARROW_ANGLE): "up",
                        (270 - LEG_ARROW_ANGLE, -90): "left",
                        (270 - 2*LEG_ARROW_ANGLE, -90): "down",
                    }
                    leg_arrow = leg_arrow_angles.get((legL_angle, legR_angle), '')
                    if leg_arrow:
                        state.current_semaphore = leg_arrow + ' arrow'
                        state.last_frames[-1]['signed'] = True

            # shift: both hands open
            shift_on = False # Disengaja dimatikan dulu sesuai file asli
            min_finger_reach = FINGER_MOUTH_RATIO * mouth_width
            palmL, palmR = body[17], body[18]
            # Untuk implementasi hand open, Anda perlu mendapatkan landmark tangan dari 'hands'
            # Ini memerlukan mapping dari index landmark tangan MediaPipe ke `hands` list.
            # Saat ini 'hands' hanya berisi koordinat, bukan landmark spesifik.
            # Anda perlu mendefinisikan struktur `hands` lebih lanjut atau langsung gunakan hand_results.multi_hand_landmarks
            # for hand_idx, hand_lm in enumerate(hand_results.multi_hand_landmarks):
            #    if hand_idx == 0: # Ini untuk tangan pertama yang terdeteksi
            #       thumb = {'x': 1-hand_lm.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x, 'y': 1-hand_lm.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y}
            #       forefinger = {'x': 1-hand_lm.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x, 'y': 1-hand_lm.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y}
            #       pinky = {'x': 1-hand_lm.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].x, 'y': 1-hand_lm.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].y}
            #       if is_hand_open(thumb, forefinger, pinky, palmL, palmR, min_finger_reach):
            #           shift_on = True


            # numbers: squat
            kneeL, kneeR = body[25], body[26]
            hipL, hipR = body[23], body[24]
            numerals = is_squatting(hipL, kneeL, hipR, kneeR)

            # alphanumeric: arm flags
            if is_limb_pointing(*armL) and is_limb_pointing(*armR):
                armL_angle = get_limb_direction(armL)
                armR_angle = get_limb_direction(armR)
                semaphore_result = SEMAPHORES.get((armL_angle, armR_angle), None)
                if semaphore_result:
                    state.current_semaphore = semaphore_result.get('n', '') if numerals else semaphore_result.get('a', '')
                    if state.current_semaphore: # Hanya menandai jika ada semaphore valid
                        state.last_frames[-1]['signed'] = True
                        state.type_and_remember_session(image, shift_on, numerals, command_on, control_on, True) # Menggunakan instance method

            # repeat last: jump (hips rise + fall)
            if is_jumping(hipL, hipR):
                if state.last_keys: # Hanya ulangi jika ada last_keys
                    state.current_semaphore = state.last_keys[-1]
                    state.last_frames[-1]['signed'] = True
                    state.type_and_remember_session(image, shift_on, numerals, command_on, control_on, True) # Menggunakan instance method
            
    # Prepare the output image for Streamlit
    # Draw landmarks and connections on the image
    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            DEFAULT_LANDMARKS_STYLE)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                DEFAULT_LANDMARKS_STYLE,
                DEFAULT_HAND_CONNECTIONS_STYLE)

    # Tambahkan overlay teks yang dibutuhkan (semaphore yang terdeteksi)
    if state.current_semaphore:
        to_display = get_key_text(state.last_keys if state.last_keys else [state.current_semaphore])
        # Gunakan state.frame_midpoint yang sudah diupdate dari dimensi gambar
        cv2.putText(image, to_display, state.frame_midpoint,
                    cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 10)


    # Encode processed image to base64
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # Prepare game state for frontend
    game_state_data = {
        "player_stats": state.player_stats,
        "current_difficulty": state.current_difficulty,
        "start_time": state.start_time,
        "game_mode": state.game_mode,
        "challenge_words": state.challenge_words,
        "challenge_index": state.challenge_index,
        "time_limit": state.time_limit,
        "words_completed_in_session": state.words_completed_in_session,
        "target_word": state.target_word,
        "typed_word": state.typed_word,
        "current_semaphore": state.current_semaphore,
        "last_keys": state.last_keys,
        "display_message": "" # Pesan untuk ditampilkan (misal: "Benar!", "Salah!")
    }

    # Tambahkan pesan berdasarkan hasil game
    if state.current_semaphore == "space":
        if state.typed_word == state.target_word:
            game_state_data["display_message"] = "✅ Benar!"
        else:
            game_state_data["display_message"] = "❌ Salah! Coba lagi."
    elif state.current_semaphore == "backspace":
        game_state_data["display_message"] = "Backspace ditekan"


    return {"image": encoded_image, "game_state": game_state_data}


@app.post("/game_action/{session_id}")
async def game_action(session_id: str, action: dict):
    if session_id not in session_states:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = session_states[session_id]
    action_type = action.get("type")
    value = action.get("value")

    if action_type == "set_difficulty":
        state.current_difficulty = value
        if state.game_mode != "challenge":
            state.target_word = state.get_random_word_session(state.current_difficulty)
            state.start_time = time.time()
        print(f"[{session_id}] 🎮 Kesulitan: {state.current_difficulty} | Kata baru: {state.target_word}")
        return {"status": "success", "message": f"Difficulty set to {value}"}
    
    elif action_type == "start_challenge_selection":
        state.game_mode = "selecting_challenge_difficulty"
        print(f"[{session_id}] 🎯 Pilih tingkat kesulitan Challenge:")
        return {"status": "success", "message": "Selecting challenge difficulty"}
    
    elif action_type == "start_challenge":
        state.current_difficulty = value # value should be "mudah", "sedang", "sulit"
        state.game_mode = "challenge"
        state.generate_challenge_session()
        state.target_word = state.challenge_words[0] if state.challenge_words else ""
        state.start_time = time.time()
        print(f"[{session_id}] 🎮 Challenge Mode dimulai! Kesulitan: {state.current_difficulty} | Kata pertama: {state.target_word}")
        return {"status": "success", "message": "Challenge mode started"}
    
    elif action_type == "start_practice":
        if not state.target_word:
            state.game_mode = "practice"
            state.target_word = state.get_random_word_session(state.current_difficulty)
            state.start_time = time.time()
            print(f"[{session_id}] 🎮 Mode: Practice | Kata: {state.target_word}")
        return {"status": "success", "message": "Practice mode started"}
    
    elif action_type == "get_stats":
        return {"status": "success", "player_stats": state.player_stats}
    
    elif action_type == "reset_game": # Tambahkan fungsi reset
        session_states[session_id] = GameState(session_id)
        print(f"[{session_id}] Game reset.")
        return {"status": "success", "message": "Game reset."}

    return {"status": "error", "message": "Unknown action"}
