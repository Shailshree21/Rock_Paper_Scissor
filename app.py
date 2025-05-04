from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import random
import time
import json
from collections import deque
from enum import Enum

app = Flask(__name__)

# Load model and initialize detector
cnn_model = tf.keras.models.load_model('model/rock_paper_scissors_model.h5')
class_names = ['rock', 'paper', 'scissors']

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
detector = HandDetector(maxHands=1, detectionCon=0.8)

# Game states enum
class GamePhase(Enum):
    WAITING = "waiting"
    COUNTDOWN = "countdown"
    SHOW_MOVE = "show_move"
    RESULT = "result"
    GAME_OVER = "game_over"

# Game state variables
class GameState:
    def __init__(self):
        self.scores = [0, 0]  # [AI, Player]
        self.game_active = False
        self.result_text = "Get Ready!"
        self.ai_move = ""
        self.player_move = ""
        self.round_time = 3  # seconds for each move
        self.result_time = 1.5  # seconds to show result
        self.round_counter = 0
        self.game_over = False
        self.move_history = deque(maxlen=5)
        self.confidence_threshold = 0.7
        self.timer_start = time.time()
        self.showing_result = False
        self.last_valid_move = None
        self.moves_this_round = set()

    def reset_round(self):
        self.last_valid_move = None
        self.moves_this_round.clear()
        self.timer_start = time.time()
        self.showing_result = False
        self.player_move = ""
        self.ai_move = ""
        self.result_text = "Show your move!"

game_state = GameState()

# Enhanced Q-learning with state history
class QLearning:
    def __init__(self):
        self.Q_table = np.zeros((3, 3, 3))  # (current_state, prev_state, action)
        self.epsilon = 0.2
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.move_to_num = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.num_to_move = {0: 'rock', 1: 'paper', 2: 'scissors'}
        
    def get_winning_move(self, move):
        return self.num_to_move[(self.move_to_num[move] + 1) % 3]
    
    def get_state(self, moves):
        if not moves:
            return random.randint(0, 2)
        return self.move_to_num[moves[-1]]
    
    def predict_move(self, player_moves):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(class_names)
        
        current_state = self.get_state(player_moves)
        prev_state = self.get_state(list(player_moves)[:-1] if len(player_moves) > 1 else [])
        
        # Pattern detection
        if len(player_moves) >= 3:
            pattern = [self.move_to_num[m] for m in player_moves[-3:]]
            if pattern[0] == pattern[1] == pattern[2]:
                return self.get_winning_move(player_moves[-1])
        
        action = np.argmax(self.Q_table[current_state, prev_state])
        return self.num_to_move[action]
    
    def update(self, state, prev_state, action, reward, next_state):
        action_idx = self.move_to_num[action]
        self.Q_table[state, prev_state, action_idx] += self.learning_rate * (
            reward + self.discount_factor * np.max(self.Q_table[next_state, state]) -
            self.Q_table[state, prev_state, action_idx]
        )

q_learning = QLearning()

# Enhanced gesture prediction with confidence threshold
def predict_gesture(imgCrop):
    imgCrop = cv2.resize(imgCrop, (224, 224))
    imgCrop = imgCrop / 255.0
    predictions = cnn_model.predict(np.expand_dims(imgCrop, axis=0), verbose=0)
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]
    
    if confidence < game_state.confidence_threshold:
        return None
    
    return predicted_class

def evaluate_round(player_move, ai_move):
    if player_move == ai_move:
        return "Draw!", 0
    
    winning_moves = {
        'rock': 'scissors',
        'paper': 'rock',
        'scissors': 'paper'
    }
    
    if winning_moves[player_move] == ai_move:
        return "Player Wins!", -1
    return "AI Wins!", 1

def update_game_state(player_move):
    if not player_move or player_move in game_state.moves_this_round:
        return
    game_state.moves_this_round.add(player_move)
    game_state.move_history.append(player_move)
    game_state.last_valid_move = player_move
    # Get AI move using enhanced Q-learning
    current_state = q_learning.get_state(list(game_state.move_history))
    prev_state = q_learning.get_state(list(game_state.move_history)[:-1] if len(game_state.move_history) > 1 else [])
    game_state.ai_move = q_learning.predict_move(game_state.move_history)
    game_state.player_move = player_move
    # Evaluate round and update scores
    result_text, reward = evaluate_round(player_move, game_state.ai_move)
    game_state.result_text = result_text
    if reward == -1:
        game_state.scores[1] += 1
    elif reward == 1:
        game_state.scores[0] += 1
    # Update Q-learning
    next_state = q_learning.get_state(list(game_state.move_history))
    q_learning.update(current_state, prev_state, game_state.ai_move, reward, next_state)
    # Update round counter and check game over
    game_state.round_counter += 1
    if game_state.round_counter >= 3:  # Best of 3 rounds
        game_state.game_over = True
        if game_state.scores[0] > game_state.scores[1]:
            game_state.result_text = "AI Wins the Game!"
        elif game_state.scores[1] > game_state.scores[0]:
            game_state.result_text = "You Win the Game!"
        else:
            game_state.result_text = "Game Draw!"
        game_state.showing_result = True
        game_state.timer_start = time.time()
    else:
        game_state.showing_result = True
        game_state.timer_start = time.time()

def generate_frames():
    last_predicted_gesture = None
    consecutive_failures = 0
    MAX_FAILURES = 10
    try:
        while True:
            try:
                success, img = cap.read()
                if not success:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_FAILURES:
                        cap.release()
                        time.sleep(1)
                        cap.open(0)
                        consecutive_failures = 0
                    time.sleep(0.05)
                    continue
                consecutive_failures = 0
                img = cv2.flip(img, 1)
                hands, img = detector.findHands(img, flipType=False)
                current_predicted_gesture = None
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    imgCrop = img[y:y+h, x:x+w]
                    if imgCrop.size != 0:
                        current_predicted_gesture = predict_gesture(imgCrop)
                # Always show the current predicted gesture for feedback
                if current_predicted_gesture:
                    last_predicted_gesture = current_predicted_gesture
                    cv2.putText(img, f"Gesture: {current_predicted_gesture}", (20, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
                else:
                    if last_predicted_gesture:
                        cv2.putText(img, f"Gesture: {last_predicted_gesture}", (20, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 1)
                # --- Single timer per move logic ---
                if game_state.game_active and not game_state.game_over:
                    elapsed = time.time() - game_state.timer_start
                    if not game_state.showing_result:
                        # Player's move phase
                        game_state.result_text = "Show your move!"
                        if current_predicted_gesture:
                            update_game_state(current_predicted_gesture)
                        elif elapsed >= game_state.round_time:
                            # No move detected, count as loss for player
                            update_game_state('no_move')
                            game_state.result_text = "No Move Detected! AI Wins the Round!"
                        if game_state.showing_result:
                            game_state.timer_start = time.time()
                    else:
                        # Showing result phase
                        if elapsed >= game_state.result_time:
                            if not game_state.game_over:
                                game_state.reset_round()
                cv2.putText(img, game_state.result_text, (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if game_state.ai_move and (game_state.showing_result or game_state.game_over):
                    ai_img_path = f'static/{game_state.ai_move}.png'
                    try:
                        ai_img = cv2.imread(ai_img_path, cv2.IMREAD_UNCHANGED)
                        if ai_img is not None:
                            ai_img = cv2.resize(ai_img, (100, 100))
                            x_offset = img.shape[1] - 120
                            y_offset = 20
                            if ai_img.shape[2] == 4:
                                alpha_channel = ai_img[:, :, 3] / 255.0
                                for c in range(3):
                                    img[y_offset:y_offset+100, x_offset:x_offset+100, c] = \
                                        (1 - alpha_channel) * img[y_offset:y_offset+100, x_offset:x_offset+100, c] + \
                                        alpha_channel * ai_img[:, :, c]
                    except Exception as e:
                        print(f"Error loading AI move image: {e}")
                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error in frame generation: {e}")
                time.sleep(0.1)
    finally:
        try:
            cap.release()
        except Exception as e:
            print(f"Error releasing camera: {e}")

def reset_game():
    global game_state
    game_state = GameState()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game')
def game():
    return render_template('game.html')

@app.route('/quit')
def quit():
    reset_game()
    return render_template('quit.html')

@app.route('/play_game', methods=['POST'])
def play_game():
    reset_game()
    game_state.game_active = True
    return '', 204

@app.route('/quit_game', methods=['POST'])
def quit_game():
    reset_game()
    return '', 204

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/game_data')
def game_data():
    if game_state.game_over or game_state.showing_result:
        ai_image = f'/static/{game_state.ai_move}.png' if game_state.ai_move else ''
    else:
        ai_image = get_random_ai_move_image()
    return jsonify({
        'ai_image': ai_image,
        'player_score': game_state.scores[1],
        'ai_score': game_state.scores[0],
        'result': game_state.result_text,
        'timer': max(0, int(game_state.round_time - (time.time() - game_state.timer_start))) if not game_state.showing_result and not game_state.game_over else 0,
        'game_over': game_state.game_over,
        'winner_message': game_state.result_text if game_state.game_over else "",
        'move_history': list(game_state.move_history),
        'current_phase': 'result' if game_state.showing_result else ('game_over' if game_state.game_over else 'move'),
    })

@app.route('/next_round', methods=['POST'])
def next_round():
    if not game_state.game_over:
        game_state.reset_round()
    return '', 204

def get_random_ai_move_image():
    move = random.choice(class_names)
    return f'/static/{move}.png'

if __name__ == '__main__':
    app.run(debug=True)
