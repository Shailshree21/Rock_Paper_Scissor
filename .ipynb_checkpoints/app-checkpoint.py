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
cnn_model = tf.keras.models.load_model('model/cnn_rps_model.h5')
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
        self.round_time = 3
        self.round_counter = 0
        self.game_over = False
        self.move_history = deque(maxlen=5)
        self.confidence_threshold = 0.7
        self.phase = GamePhase.WAITING
        self.phase_start_time = time.time()
        self.countdown_duration = 3
        self.show_move_duration = 2
        self.result_duration = 2
        self.last_valid_move = None
        self.moves_this_round = set()

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
        game_state.phase = GamePhase.GAME_OVER
    else:
        game_state.phase = GamePhase.RESULT
        game_state.phase_start_time = time.time()

def handle_game_phases():
    current_time = time.time()
    elapsed_time = current_time - game_state.phase_start_time

    if game_state.phase == GamePhase.WAITING:
        game_state.result_text = "Get Ready!"
        game_state.phase = GamePhase.COUNTDOWN
        game_state.phase_start_time = current_time
        game_state.moves_this_round.clear()
        
    elif game_state.phase == GamePhase.COUNTDOWN:
        remaining = game_state.countdown_duration - elapsed_time
        if remaining <= 0:
            game_state.phase = GamePhase.SHOW_MOVE
            game_state.phase_start_time = current_time
            game_state.result_text = "Show your move!"
        else:
            game_state.result_text = f"Get Ready: {int(remaining)}"
            
    elif game_state.phase == GamePhase.SHOW_MOVE:
        if elapsed_time >= game_state.show_move_duration:
            if game_state.last_valid_move:
                game_state.phase = GamePhase.RESULT
                game_state.phase_start_time = current_time
            else:
                # No valid move detected, restart countdown
                game_state.phase = GamePhase.COUNTDOWN
                game_state.phase_start_time = current_time
                game_state.result_text = "No move detected! Try again!"
                
    elif game_state.phase == GamePhase.RESULT:
        if elapsed_time >= game_state.result_duration:
            if not game_state.game_over:
                game_state.phase = GamePhase.WAITING
                game_state.phase_start_time = current_time
                game_state.last_valid_move = None

def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=False)

        if game_state.game_active and not game_state.game_over:
            handle_game_phases()
            
            if game_state.phase == GamePhase.SHOW_MOVE and hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgCrop = img[y:y+h, x:x+w]
                
                if imgCrop.size != 0:
                    player_move = predict_gesture(imgCrop)
                    if player_move:
                        update_game_state(player_move)

        # Add visual elements to the frame
        cv2.putText(img, game_state.result_text, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if game_state.ai_move and game_state.phase in [GamePhase.RESULT, GamePhase.GAME_OVER]:
            # Load and overlay AI move image
            ai_img_path = f'static/{game_state.ai_move}.png'
            try:
                ai_img = cv2.imread(ai_img_path, cv2.IMREAD_UNCHANGED)
                if ai_img is not None:
                    ai_img = cv2.resize(ai_img, (100, 100))
                    x_offset = img.shape[1] - 120
                    y_offset = 20
                    if ai_img.shape[2] == 4:  # If PNG with alpha channel
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
    ai_image = f'/static/{game_state.ai_move}.png' if game_state.ai_move else ''
    
    return jsonify({
        'ai_image': ai_image,
        'player_score': game_state.scores[1],
        'ai_score': game_state.scores[0],
        'result': game_state.result_text,
        'timer': game_state.round_time,
        'game_over': game_state.game_over,
        'winner_message': game_state.result_text if game_state.game_over else "",
        'move_history': list(game_state.move_history),
        'current_phase': game_state.phase.value
    })

if __name__ == '__main__':
    app.run(debug=True)
