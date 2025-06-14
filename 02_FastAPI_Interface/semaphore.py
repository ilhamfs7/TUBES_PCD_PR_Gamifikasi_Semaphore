import argparse
# from keyboard import keyboard  # local fork
import keyboard  # standard PyPI version

import mediapipe as mp
import cv2
import random
import time
import json
import os

from scipy.spatial import distance as dist
from math import atan, atan2, pi, degrees
from datetime import datetime

# run pake body_landmark = python semaphore.py --landmarks yes

DEFAULT_LANDMARKS_STYLE = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
DEFAULT_HAND_CONNECTIONS_STYLE = mp.solutions.drawing_styles.get_default_hand_connections_style()

# Optionally record the video feed to a timestamped AVI in the current directory
RECORDING_FILENAME = datetime.now().strftime('%Y%m%d_%H%M%S') + '.avi'  # Safer filename format
FPS = 10

VISIBILITY_THRESHOLD = .8 # amount of certainty that a body landmark is visible
STRAIGHT_LIMB_MARGIN = 20 # degrees from 180
EXTENDED_LIMB_MARGIN = .8 # lower limb length as fraction of upper limb

LEG_LIFT_MIN = -30 # degrees below horizontal

ARM_CROSSED_RATIO = 2 # max distance from wrist to opposite elbow, relative to mouth width

MOUTH_COVER_THRESHOLD = .03 # hands over mouth max distance error out of 1

SQUAT_THRESHOLD = .1 # max hip-to-knee vertical distance

JUMP_THRESHOLD = .0001

LEG_ARROW_ANGLE = 18 # degrees from vertical standing; should be divisor of 90

FINGER_MOUTH_RATIO = 1.5 # open hand relative to mouth width

# R side: 90 top to 0 right to -90 bottom
# L side: 90 top to 180 left to 269... -> -90 bottom
SEMAPHORES = {
    (-90, -45): {'a': "a", 'n': "1"},
    (-90, 0): {'a': "b", 'n': "2"},
    (-90, 45): {'a': "c", 'n': "3"},
    (-90, 90): {'a': "d", 'n': "4"},
    (135, -90): {'a': "e", 'n': "5"},
    (180, -90): {'a': "f", 'n': "6"},
    (225, -90): {'a': "g", 'n': "7"},
    (-45, 0): {'a': "h", 'n': "8"},
    (-45, 45): {'a': "i", 'n': "9"},
    (180, 90): {'a': "j", 'n': "capslock"},
    (90, -45): {'a': "k", 'n': "0"},
    (135, -45): {'a': "l", 'n': "\\"},
    (180, -45): {'a': "m", 'n': "["},
    (225, -45): {'a': "n", 'n': "]"},
    (0, 45): {'a': "o", 'n': ","},
    (90, 0): {'a': "p", 'n': ";"},
    (135, 0): {'a': "q", 'n': "="},
    (180, 0): {'a': "r", 'n': "-"},
    (225, 0): {'a': "s", 'n': "."},
    (90, 45): {'a': "t", 'n': "`"},
    (135, 45): {'a': "u", 'n': "/"},
    (225, 90): {'a': "v", 'n': '"'},
    (135, 180): {'a': "w"},
    (135, 225): {'a': "x", 'n': ""}, # clear last signal
    (180, 45): {'a': "y"},
    (180, 225): {'a': "z"},
    (90, 90): {'a': "space", 'n': "enter"},
    # (135, 90): {'a': "tab"}, # custom "numerals" replacement
    # (225, 45): {'a': "escape"}, # custom "cancel" replacement
}

leg_arrow_angles = {
    (-90, -90 + LEG_ARROW_ANGLE): "right",
    (-90, -90 + 2*LEG_ARROW_ANGLE): "up",
    (270 - LEG_ARROW_ANGLE, -90): "left",
    (270 - 2*LEG_ARROW_ANGLE, -90): "down",
}

FRAME_HISTORY = 8 # pose history is compared against FRAME_HISTORY recent frames
HALF_HISTORY = int(FRAME_HISTORY/2)
QUARTER_HISTORY = int(FRAME_HISTORY/4)

empty_frame = {
    'hipL_y': 0,
    'hipR_y': 0,
    'hips_dy': 0,
    'dxL_thrust_hipL': 0,
    'dxL_thrust_hipR': 0,
    'dxR_thrust_hipL': 0,
    'dxR_thrust_hipR': 0,
    'signed': False,
}
last_frames = FRAME_HISTORY*[empty_frame.copy()]

frame_midpoint = (0,0)

current_semaphore = ''
last_keys = []
typed_word = ""
target_word = ""

# Sistem Gamifikasi
player_stats = {
    "level": 1,
    "experience": 0,
    "streak": 0,
    "max_streak": 0,
    "total_words": 0,
    "accuracy": 100.0,
    "achievements": [],
    "coins": 0
}

# Level system
LEVEL_REQUIREMENTS = [0, 100, 250, 500, 1000, 2000, 3500, 5500, 8000, 12000, 17000]
EXPERIENCE_PER_WORD = 50
STREAK_BONUS = 10
ACCURACY_BONUS = 25

# Daftar kata dengan tingkat kesulitan
WORD_LISTS = {
    "mudah": ["aku", "dia", "kamu", "saya", "kami", "mama", "papa", "air", "api", "mau"],
    "sedang": ["sekolah", "rumah", "teman", "belajar", "main", "makan", "minum", "tidur", "bangun", "kerja"],
    "sulit": ["semaphore", "komunikasi", "teknologi", "programming", "artificial", "intelligence", "komputer", "keyboard", "monitor", "internet"]
}

# Achievement system
ACHIEVEMENTS = {
    "first_word": {"name": "Kata Pertama", "desc": "Selesaikan kata pertama", "reward": 50},
    "streak_5": {"name": "Rajin Berlatih", "desc": "Capai streak 5", "reward": 100},
    "streak_10": {"name": "Semangat Tinggi", "desc": "Capai streak 10", "reward": 200},
    "accuracy_95": {"name": "Hampir Sempurna", "desc": "Capai akurasi 95%", "reward": 150},
    "accuracy_100": {"name": "Sempurna!", "desc": "Capai akurasi 100%", "reward": 300},
    "level_5": {"name": "Ahli Pemula", "desc": "Capai level 5", "reward": 250},
    "level_10": {"name": "Master Semaphore", "desc": "Capai level 10", "reward": 500},
    "speed_demon": {"name": "Kilat", "desc": "Selesaikan kata dalam 10 detik", "reward": 100}
}

# Game state
current_difficulty = "mudah"
start_time = None
game_mode = "practice"  # practice, challenge, time_attack
challenge_words = []
challenge_index = 0
time_limit = 60  # untuk mode time attack
words_completed_in_session = 0
last_semaphore = None  # Tambahan untuk melacak huruf terakhir
last_semaphore_time = 0  # Tambahan untuk melacak waktu deteksi terakhir
COOLDOWN_TIME = 1.0

def get_angle(a, b, c):
    ang = degrees(atan2(c['y']-b['y'], c['x']-b['x']) - atan2(a['y']-b['y'], a['x']-b['x']))
    return ang + 360 if ang < 0 else ang

def is_missing(part):
    return any(joint['visibility'] < VISIBILITY_THRESHOLD for joint in part)

def is_limb_pointing(upper, mid, lower):
    if is_missing([upper, mid, lower]):
        return False
    limb_angle = get_angle(upper, mid, lower)
    is_in_line = abs(180 - limb_angle) < STRAIGHT_LIMB_MARGIN
    if is_in_line:
        upper_length = dist.euclidean([upper['x'], upper['y']], [mid['x'], mid['y']])
        lower_length = dist.euclidean([lower['x'], lower['y']], [mid['x'], mid['y']])
        is_extended = lower_length > EXTENDED_LIMB_MARGIN * upper_length
        return is_extended
    return False

def get_limb_direction(arm, closest_degrees=45):
    # should also use atan2 but I don't want to do more math
    dy = arm[2]['y'] - arm[0]['y'] # wrist -> shoulder
    dx = arm[2]['x'] - arm[0]['x']
    angle = degrees(atan(dy/dx))
    if (dx < 0):
        angle += 180

    # collapse to nearest closest_degrees; 45 for semaphore
    mod_close = angle % closest_degrees
    angle -= mod_close
    if mod_close > closest_degrees/2:
        angle += closest_degrees

    angle = int(angle)
    if angle == 270:
        angle = -90

    return angle

def is_arm_crossed(elbow, wrist, max_dist):
    return dist.euclidean([elbow['x'], elbow['y']], [wrist['x'], wrist['y']]) < max_dist

def is_arms_crossed(elbowL, wristL, elbowR, wristR, mouth_width):
    max_dist = mouth_width * ARM_CROSSED_RATIO
    return is_arm_crossed(elbowL, wristR, max_dist) and is_arm_crossed(elbowR, wristL, max_dist)

def is_leg_lifted(leg):
    if is_missing(leg):
        return False
    dy = leg[1]['y'] - leg[0]['y'] # knee -> hip
    dx = leg[1]['x'] - leg[0]['x']
    angle = degrees(atan2(dy, dx))
    return angle > LEG_LIFT_MIN

def is_jumping(hipL, hipR):
    global last_frames

    if is_missing([hipL, hipR]):
        return False

    last_frames[-1]['hipL_y'] = hipL['y']
    last_frames[-1]['hipR_y'] = hipR['y']

    if (hipL['y'] > last_frames[-2]['hipL_y'] + JUMP_THRESHOLD) and (
        hipR['y'] > last_frames[-2]['hipR_y'] + JUMP_THRESHOLD):
        last_frames[-1]['hips_dy'] = 1 # rising
    elif (hipL['y'] < last_frames[-2]['hipL_y'] - JUMP_THRESHOLD) and (
            hipR['y'] < last_frames[-2]['hipR_y'] - JUMP_THRESHOLD):
        last_frames[-1]['hips_dy'] = -1 # falling
    else:
        last_frames[-1]['hips_dy'] = 0 # not significant dy

    # consistently rising first half, lowering second half
    jump_up = all(frame['hips_dy'] == 1 for frame in last_frames[:HALF_HISTORY])
    get_down = all(frame['hips_dy'] == -1 for frame in last_frames[HALF_HISTORY:])

    return jump_up and get_down

def is_mouth_covered(mouth, palms):
    if is_missing(palms):
        return False
    dxL = (mouth[0]['x'] - palms[0]['x'])
    dyL = (mouth[0]['y'] - palms[0]['y'])
    dxR = (mouth[1]['x'] - palms[1]['x'])
    dyR = (mouth[1]['y'] - palms[1]['y'])
    return all(abs(d) < MOUTH_COVER_THRESHOLD for d in [dxL, dyL, dxR, dyR])

def is_squatting(hipL, kneeL, hipR, kneeR):
    if is_missing([hipL, kneeL, hipR, kneeR]):
        return False
    dyL = abs(hipL['y'] - kneeL['y'])
    dyR = abs(hipR['y'] - kneeR['y'])
    return (dyL < SQUAT_THRESHOLD) and (dyR < SQUAT_THRESHOLD)

def is_finger_out(finger, palmL, palmR, min_finger_reach):
    dL_finger = dist.euclidean([finger['x'], finger['y']], [palmL['x'], palmL['y']])
    dR_finger = dist.euclidean([finger['x'], finger['y']], [palmR['x'], palmR['y']])
    d_finger = min(dL_finger, dR_finger)
    return d_finger > min_finger_reach

def is_hand_open(thumb, forefinger, pinky, palmL, palmR, min_finger_reach):
    thumb_out = is_finger_out(thumb, palmL, palmR, min_finger_reach)
    forefinger_out = is_finger_out(forefinger, palmL, palmR, min_finger_reach)
    pinky_out = is_finger_out(pinky, palmL, palmR, min_finger_reach)
    return thumb_out and forefinger_out and pinky_out

def type_semaphore(armL_angle, armR_angle, image, shift_on, numerals, command_on, control_on, display_only, allow_repeat):
    global current_semaphore, last_semaphore, last_semaphore_time
    arm_match = SEMAPHORES.get((armL_angle, armR_angle), '')
    if arm_match:
        new_semaphore = arm_match.get('n', '') if numerals else arm_match.get('a', '')
        current_time = time.time()
        # Cek apakah huruf sama dengan huruf terakhir
        if allow_repeat and new_semaphore == last_semaphore and (current_time - last_semaphore_time) < COOLDOWN_TIME:
            return False  # Skip jika huruf sama dan belum lewat cooldown
        current_semaphore = new_semaphore
        last_semaphore = current_semaphore
        last_semaphore_time = current_time
        type_and_remember(image, shift_on, command_on, control_on, display_only, allow_repeat)
        return current_semaphore
    return False

def type_and_remember(image=None, shift_on=False, command_on=False, control_on=False, display_only=True, allow_repeat=False):
    global current_semaphore, last_keys, typed_word, target_word, start_time, words_completed_in_session, game_mode, challenge_words, challenge_index
    if len(current_semaphore) == 0 or not target_word:
        return
    keys = []
    if shift_on:
        keys.append('shift')
    if command_on:
        keys.append('command')
    if control_on:
        keys.append('control')
    keys.append(current_semaphore)
    if allow_repeat or (keys != last_keys):
        last_keys = keys.copy()
        if current_semaphore == "space":
            word_time = time.time() - start_time if start_time else 0
            if typed_word == target_word:
                player_stats["total_words"] += 1
                player_stats["streak"] += 1
                words_completed_in_session += 1
                if player_stats["streak"] > player_stats["max_streak"]:
                    player_stats["max_streak"] = player_stats["streak"]
                base_exp = EXPERIENCE_PER_WORD
                streak_bonus = min(player_stats["streak"] * STREAK_BONUS, 200)
                speed_bonus = max(0, 50 - int(word_time)) if word_time > 0 else 0
                total_exp = base_exp + streak_bonus + speed_bonus
                add_experience(total_exp)
                player_stats["coins"] += 10 + (player_stats["streak"] // 5)
                if player_stats["total_words"] == 1:
                    check_achievement("first_word")
                if player_stats["streak"] == 5:
                    check_achievement("streak_5")
                elif player_stats["streak"] == 10:
                    check_achievement("streak_10")
                if word_time <= 10 and word_time > 0:
                    check_achievement("speed_demon")
                player_stats["accuracy"] = calculate_accuracy()
                if player_stats["accuracy"] >= 95:
                    check_achievement("accuracy_95")
                if player_stats["accuracy"] >= 100:
                    check_achievement("accuracy_100")
                if game_mode == "challenge":
                    challenge_index += 1
                    if challenge_index < len(challenge_words):
                        target_word = challenge_words[challenge_index]
                    else:
                        game_mode = "practice"
                        target_word = get_random_word(current_difficulty)
                        add_experience(100)
                else:
                    target_word = get_random_word(current_difficulty)
            else:
                player_stats["streak"] = 0
                player_stats["total_words"] += 1
                player_stats["accuracy"] = calculate_accuracy()
            typed_word = ""
            start_time = time.time()
            save_player_stats()
        else:
            typed_word += current_semaphore
            if not start_time:
                start_time = time.time()
        current_semaphore = ''
        output(keys, image, display_only)

def get_key_text(keys):
    if not (len(keys) > 0):
        return ''

    semaphore = keys[-1]
    keystring = ''
    if 'shift' in keys:
        keystring += 'S+'
    if 'command' in keys:
        keystring += 'CMD+'
    if 'control' in keys:
        keystring += 'CTL+'

    keystring += semaphore
    return keystring

def load_player_stats():
    """Load player statistics from file"""
    global player_stats
    try:
        if os.path.exists("semaphore_stats.json"):
            with open("semaphore_stats.json", "r") as f:
                player_stats = json.load(f)
    except Exception as e:
        print(f"Error loading player stats: {e}")

def save_player_stats():
    """Save player statistics to file"""
    try:
        with open("semaphore_stats.json", "w") as f:
            json.dump(player_stats, f, indent=2)
    except Exception as e:
        print(f"Error saving player stats: {e}")

def get_random_word(difficulty="mudah"):
    """Get random word based on difficulty"""
    return random.choice(WORD_LISTS[difficulty])

def calculate_accuracy():
    """Calculate typing accuracy"""
    if player_stats["total_words"] == 0:
        return 100.0
    return (player_stats["streak"] / player_stats["total_words"]) * 100

def add_experience(amount):
    """Add experience and check for level up"""
    global player_stats
    player_stats["experience"] += amount
    
    # Check for level up
    current_level = player_stats["level"]
    while (current_level < len(LEVEL_REQUIREMENTS) - 1 and 
           player_stats["experience"] >= LEVEL_REQUIREMENTS[current_level]):
        current_level += 1
        player_stats["level"] = current_level
        player_stats["coins"] += 100 * current_level
        print(f"🎉 LEVEL UP! Sekarang level {current_level}!")
        check_achievement(f"level_{current_level}")

def check_achievement(achievement_id):
    """Check and award achievements"""
    global player_stats
    
    if achievement_id in ACHIEVEMENTS and achievement_id not in player_stats["achievements"]:
        achievement = ACHIEVEMENTS[achievement_id]
        player_stats["achievements"].append(achievement_id)
        player_stats["coins"] += achievement["reward"]
        print(f"🏆 ACHIEVEMENT UNLOCKED: {achievement['name']}")
        print(f"   {achievement['desc']} (+{achievement['reward']} coins)")

def display_stats(image):
    """Display player stats on screen"""
    stats_text = [
        f"Level: {player_stats['level']} | XP: {player_stats['experience']}",
        f"Streak: {player_stats['streak']} | Max: {player_stats['max_streak']}",
        f"Akurasi: {player_stats['accuracy']:.1f}% | Coins: {player_stats['coins']}",
        f"Mode: {game_mode.upper()} | Kesulitan: {current_difficulty.upper()}"
    ]
    
    y_pos = 30
    for text in stats_text:
        cv2.putText(image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25

def display_word_challenge(image):
    """Display current word challenge"""
    global target_word, typed_word, start_time
    
    # Jangan tampilkan jika belum ada target word
    if not target_word:
        cv2.putText(image, "Tekan P untuk Practice atau C untuk Challenge", 
                    (10, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return

    # Display target word
    cv2.putText(image, f"Target: {target_word}", (10, image.shape[0] - 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Display typed word
    cv2.putText(image, f"Typed: {typed_word}", (10, image.shape[0] - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    
    # Display timer for time attack mode
    if game_mode == "time_attack" and start_time:
        elapsed = time.time() - start_time
        remaining = max(0, time_limit - elapsed)
        cv2.putText(image, f"Time: {remaining:.1f}s", (image.shape[1] - 200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

def generate_challenge():
    """Generate challenge words based on difficulty"""
    global challenge_words, challenge_index
    
    num_words = 5 if current_difficulty == "mudah" else 8 if current_difficulty == "sedang" else 10
    challenge_words = [get_random_word(current_difficulty) for _ in range(num_words)]
    challenge_index = 0
    
    print(f"🎯 Challenge Mode: {num_words} kata tingkat {current_difficulty}")
    print(f"Kata-kata: {', '.join(challenge_words)}")

def start_game_mode(mode, difficulty):
    global game_mode, current_difficulty, target_word, typed_word, start_time, challenge_words, challenge_index
    game_mode = mode
    current_difficulty = difficulty
    typed_word = ""
    if mode == "practice":
        target_word = get_random_word(difficulty)
        start_time = time.time()
    elif mode == "challenge":
        num_words = 5 if difficulty == "mudah" else 8 if difficulty == "sedang" else 10
        challenge_words = [get_random_word(difficulty) for _ in range(num_words)]
        challenge_index = 0
        target_word = challenge_words[0]
        start_time = time.time()

def output(keys, image, display_only=True):
    keystring = '+'.join(keys)
    if len(keystring):
        # print("keys:", keystring)
        if not display_only:
            try:
                keyboard.press(keystring)
            except Exception as e:
                print(f"Error pressing keys: {e}")
        else:
            to_display = get_key_text(keys)
            cv2.putText(image, to_display, frame_midpoint,
                        cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 10)

def render_and_maybe_exit(image, recording):
    # Display game UI
    display_stats(image)
    display_word_challenge(image)
    
    cv2.imshow('Semaphore Game', image)
    if recording:
        recording.write(image)
    
    key = cv2.waitKey(5) & 0xFF
    
    # Handle keyboard shortcuts
    global current_difficulty, target_word, game_mode, start_time
    if key in [ord('1'), ord('2'), ord('3')] and game_mode == "selecting_challenge_difficulty":
        difficulty_map = {ord('1'): "mudah", ord('2'): "sedang", ord('3'): "sulit"}
        current_difficulty = difficulty_map[key]
        game_mode = "challenge"
        generate_challenge()
        target_word = challenge_words[0]
        start_time = time.time()
        print(f"🎮 Challenge Mode dimulai! Kesulitan: {current_difficulty} | Kata pertama: {target_word}")
    elif key == ord('1'):  # Change difficulty to easy
        current_difficulty = "mudah"
        if game_mode != "challenge":  # Only change target word if not in challenge
            target_word = get_random_word(current_difficulty)
            start_time = time.time()
        print(f"🎮 Kesulitan: {current_difficulty} | Kata baru: {target_word}")
    elif key == ord('2'):  # Change difficulty to medium
        current_difficulty = "sedang"
        if game_mode != "challenge":  # Only change target word if not in challenge
            target_word = get_random_word(current_difficulty)
            start_time = time.time()
        print(f"🎮 Kesulitan: {current_difficulty} | Kata baru: {target_word}")
    elif key == ord('3'):  # Change difficulty to hard
        current_difficulty = "sulit"
        if game_mode != "challenge":  # Only change target word if not in challenge
            target_word = get_random_word(current_difficulty)
            start_time = time.time()
        print(f"🎮 Kesulitan: {current_difficulty} | Kata baru: {target_word}")
    elif key == ord('c'):  # Start challenge mode
        print("🎯 Pilih tingkat kesulitan Challenge:")
        print("1 - Mudah (5 kata) | 2 - Sedang (8 kata) | 3 - Sulit (10 kata)")
        game_mode = "selecting_challenge_difficulty"
    elif key == ord('p'):  # Practice mode
        if not target_word:  # Only set if not already set
            game_mode = "practice"
            target_word = get_random_word(current_difficulty)
            start_time = time.time()
            print(f"🎮 Mode: Practice | Kata: {target_word}")
    elif key == ord('s'):  # Show stats
        print(f"\n📊 STATISTIK PEMAIN")
        print(f"Level: {player_stats['level']}")
        print(f"Experience: {player_stats['experience']}")
        print(f"Streak: {player_stats['streak']} (Max: {player_stats['max_streak']})")
        print(f"Total kata: {player_stats['total_words']}")
        print(f"Akurasi: {player_stats['accuracy']:.1f}%")
        print(f"Coins: {player_stats['coins']}")
        print(f"Achievements: {len(player_stats['achievements'])}")
    
    return key == 27  # ESC to exit

def main():
    global last_frames, frame_midpoint, target_word, start_time

    # Load player data
    load_player_stats()
    
    # Initialize game - don't start immediately
    target_word = ""
    start_time = None

    print("🎮 SEMAPHORE GAME DIMULAI!")
    print("🎮 Pilih mode game terlebih dahulu:")
    print("P - Practice Mode | C - Challenge Mode")
    print("1/2/3 - Ubah kesulitan (Mudah/Sedang/Sulit)")
    print("S - Tampilkan statistik")
    print("ESC - Keluar")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='Input video device or file (number or path), defaults to 0', default='0')
    parser.add_argument('--flip', '-f', help='Set to any value to flip resulting output (selfie view)')
    parser.add_argument('--landmarks', '-l', help='Set to any value to draw body landmarks')
    parser.add_argument('--record', '-r', help='Set to any value to save a timestamped AVI in current directory')
    parser.add_argument('--type', '-t', help='Set to any value to type output rather than only display')
    parser.add_argument('--repeat', '-p', help='Set to any value to allow instant semaphore repetitions')
    args = parser.parse_args()

    INPUT = int(args.input) if (args.input and args.input.isdigit()) else args.input
    FLIP = args.flip is not None
    DRAW_LANDMARKS = args.landmarks is not None
    RECORDING = args.record is not None
    DISPLAY_ONLY = args.type is None
    ALLOW_REPEAT = args.repeat is not None

    try:
        cap = cv2.VideoCapture(INPUT)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {INPUT}")
        
        frame_size = (int(cap.get(3)), int(cap.get(4)))
        frame_midpoint = (int(frame_size[0]/2), int(frame_size[1]/2))

        recording = None
        if RECORDING:
            try:
                recording = cv2.VideoWriter(RECORDING_FILENAME,
                                           cv2.VideoWriter_fourcc(*'MJPG'), FPS, frame_size)
                if not recording.isOpened():
                    raise ValueError(f"Cannot open video writer for {RECORDING_FILENAME}")
            except Exception as e:
                print(f"Error initializing video writer: {e}")
                recording = None

        try:
            with mp.solutions.pose.Pose() as pose_model:
                with mp.solutions.hands.Hands(max_num_hands=2) as hands_model:
                    while cap.isOpened():
                        success, image = cap.read()
                        if not success:
                            print("Failed to read frame from video source")
                            break

                        image.flags.writeable = False
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        pose_results = pose_model.process(image)
                        hand_results = hands_model.process(image)

                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        # draw pose
                        if DRAW_LANDMARKS:
                            mp.solutions.drawing_utils.draw_landmarks(
                                image,
                                pose_results.pose_landmarks,
                                mp.solutions.pose.POSE_CONNECTIONS,
                                DEFAULT_LANDMARKS_STYLE)

                        hands = []
                        hand_index = 0
                        if hand_results.multi_hand_landmarks:
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                # draw hands
                                if DRAW_LANDMARKS:
                                    mp.solutions.drawing_utils.draw_landmarks(
                                        image,
                                        hand_landmarks,
                                        mp.solutions.hands.HAND_CONNECTIONS,
                                        DEFAULT_LANDMARKS_STYLE,
                                        DEFAULT_HAND_CONNECTIONS_STYLE)
                                hands.append([])
                                for point in hand_landmarks.landmark:
                                    hands[hand_index].append({
                                        'x': 1 - point.x,
                                        'y': 1 - point.y
                                    })
                                hand_index += 1

                        if FLIP:
                            image = cv2.flip(image, 1) # selfie view

                        if pose_results.pose_landmarks:
                            # prepare to store most recent frame of movement updates over time
                            last_frames = last_frames[1:] + [empty_frame.copy()]

                            # short cool off period of last_frames for each sign
                            if any(frame['signed'] for frame in last_frames):
                                if render_and_maybe_exit(image, recording):
                                    break
                                else:
                                    continue

                            body = []
                            # (0,0) bottom left to (1,1) top right
                            for point in pose_results.pose_landmarks.landmark:
                                body.append({
                                    'x': 1 - point.x,
                                    'y': 1 - point.y,
                                    'visibility': point.visibility
                                })

                            # cover mouth: backspace
                            mouth = (body[9], body[10])
                            palms = (body[19], body[20])
                            if is_mouth_covered(mouth, palms):
                                output(['backspace'], image, DISPLAY_ONLY)

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
                            if is_arms_crossed(elbowL, wristL, elbowR, wristR, mouth_width):
                                if is_limb_pointing(*legL) and is_limb_pointing(*legR):
                                    legL_angle = get_limb_direction(legL, LEG_ARROW_ANGLE)
                                    legR_angle = get_limb_direction(legR, LEG_ARROW_ANGLE)
                                    leg_arrow = leg_arrow_angles.get((legL_angle, legR_angle), '')
                                    if leg_arrow:
                                        output([leg_arrow + ' arrow'], image, DISPLAY_ONLY)

                            # shift: both hands open
                            # shift_on = len(hands) > 0
                            shift_on = False # biar perintah shift ga jalan dulu
                            min_finger_reach = FINGER_MOUTH_RATIO * mouth_width
                            palmL, palmR = body[17], body[18]
                            for hand in hands:
                                thumb, forefinger, pinky = hand[4], hand[8], hand[20]
                                hand_open = is_hand_open(thumb, forefinger, pinky, palmL, palmR, min_finger_reach)
                                shift_on = shift_on and hand_open

                            # numbers: squat
                            kneeL, kneeR = body[25], body[26]
                            hipL, hipR = body[23], body[24]
                            numerals = is_squatting(hipL, kneeL, hipR, kneeR)

                            # alphanumeric: arm flags
                            if is_limb_pointing(*armL) and is_limb_pointing(*armR):
                                armL_angle = get_limb_direction(armL)
                                armR_angle = get_limb_direction(armR)
                                if type_semaphore(armL_angle, armR_angle, image,
                                                  shift_on, numerals, command_on, control_on, DISPLAY_ONLY, ALLOW_REPEAT):
                                    last_frames[-1]['signed'] = True

                            # repeat last: jump (hips rise + fall)
                            # TODO: if ankles are always in view, could be more accurate than hips
                            if is_jumping(hipL, hipR):
                                output(last_keys, image, DISPLAY_ONLY)

                        if render_and_maybe_exit(image, recording):
                            break

        finally:
            if recording:
                recording.release()
            cap.release()
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error in main loop: {e}")
        if recording:
            recording.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()