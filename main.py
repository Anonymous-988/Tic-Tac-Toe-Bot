import streamlit as st
import numpy as np
import random
import pickle
from typing import Dict, List, Tuple, Optional
import time

class TicTacToeBot:
    def __init__(self, epsilon: float = 0.1, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table: Dict[str, List[float]] = {}
        self.training_stats = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0
        }
    
    def get_state_key(self, board: np.ndarray) -> str:
        """Convert board state to string key for Q-table"""
        return ''.join(map(str, board.flatten()))
    
    def get_valid_actions(self, board: np.ndarray) -> List[int]:
        """Get list of valid actions (empty cells)"""
        return [i for i in range(9) if board.flatten()[i] == 0]
    
    def get_q_values(self, state: str) -> List[float]:
        """Get Q-values for a state, initialize if not exists"""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 9
        return self.q_table[state]
    
    def choose_action(self, board: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy strategy"""
        state = self.get_state_key(board)
        valid_actions = self.get_valid_actions(board)
        
        if not valid_actions:
            return -1
        
        # During training, use epsilon-greedy
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Greedy action selection
        q_values = self.get_q_values(state)
        valid_q_values = [(action, q_values[action]) for action in valid_actions]
        best_action = max(valid_q_values, key=lambda x: x[1])[0]
        
        return best_action
    
    def update_q_value(self, state: str, action: int, reward: float, next_state: str):
        """Update Q-value using Q-learning formula"""
        current_q = self.get_q_values(state)[action]
        next_q_values = self.get_q_values(next_state)
        max_next_q = max(next_q_values) if next_q_values else 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q

class TicTacToeGame:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the game board"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
    
    def make_move(self, position: int, player: int) -> bool:
        """Make a move on the board"""
        if self.game_over:
            return False
        
        row, col = position // 3, position % 3
        if self.board[row, col] != 0:
            return False
        
        self.board[row, col] = player
        self.winner = self.check_winner()
        if self.winner or self.is_board_full():
            self.game_over = True
        else:
            self.current_player = 3 - player  # Switch between 1 and 2
        
        return True
    
    def check_winner(self) -> Optional[int]:
        """Check if there's a winner"""
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != 0:
                return row[0]
        
        # Check columns
        for col in range(3):
            if self.board[0, col] == self.board[1, col] == self.board[2, col] != 0:
                return self.board[0, col]
        
        # Check diagonals
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
            return self.board[0, 0]
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] != 0:
            return self.board[0, 2]
        
        return None
    
    def is_board_full(self) -> bool:
        """Check if the board is full"""
        return np.all(self.board != 0)
    
    def get_reward(self, player: int) -> float:
        """Get reward for the current game state"""
        if self.winner == player:
            return 1.0
        elif self.winner == (3 - player):
            return -1.0
        elif self.is_board_full():
            return 0.0
        else:
            return 0.0

def train_bot(bot: TicTacToeBot, num_games: int, progress_bar) -> Dict:
    """Train the bot by playing games against itself"""
    training_log = []
    
    for game_num in range(num_games):
        game = TicTacToeGame()
        states_actions = []  # Store (state, action) pairs for both players
        
        # Play one game
        while not game.game_over:
            current_state = bot.get_state_key(game.board)
            action = bot.choose_action(game.board, training=True)
            
            if action == -1:  # No valid actions
                break
            
            states_actions.append((current_state, action, game.current_player))
            game.make_move(action, game.current_player)
        
        # Update Q-values based on game outcome
        for i, (state, action, player) in enumerate(states_actions):
            reward = game.get_reward(player)
            
            # Get next state (if exists)
            next_state = ""
            if i + 1 < len(states_actions):
                # Create next state by applying the next action
                temp_game = TicTacToeGame()
                temp_game.board = np.array([int(x) for x in state]).reshape(3, 3)
                temp_game.make_move(states_actions[i + 1][1], states_actions[i + 1][2])
                next_state = bot.get_state_key(temp_game.board)
            else:
                next_state = bot.get_state_key(game.board)
            
            bot.update_q_value(state, action, reward, next_state)
        
        # Update statistics
        bot.training_stats['games_played'] += 1
        if game.winner == 1:
            bot.training_stats['wins'] += 1
        elif game.winner == 2:
            bot.training_stats['losses'] += 1
        else:
            bot.training_stats['draws'] += 1
        
        # Update progress
        if game_num % (num_games // 20) == 0:
            progress = (game_num + 1) / num_games
            progress_bar.progress(progress)
            
        # Log periodic updates
        if game_num % (num_games // 10) == 0:
            win_rate = bot.training_stats['wins'] / max(bot.training_stats['games_played'], 1) * 100
            training_log.append(f"Game {game_num + 1}: Win Rate = {win_rate:.1f}%")
    
    return training_log

def display_board(board: np.ndarray):
    """Display the tic-tac-toe board using Streamlit"""
    col1, col2, col3 = st.columns(3)
    
    for i in range(3):
        for j in range(3):
            pos = i * 3 + j
            cell_value = board[i, j]
            
            if cell_value == 0:
                display_value = ""
            elif cell_value == 1:
                display_value = "❌"
            else:
                display_value = "⭕"
            
            if j == 0:
                with col1:
                    if st.button(display_value if display_value else "⬜", 
                               key=f"cell_{pos}", 
                               help=f"Position {pos}"):
                        if 'human_move' not in st.session_state:
                            st.session_state.human_move = pos
            elif j == 1:
                with col2:
                    if st.button(display_value if display_value else "⬜", 
                               key=f"cell_{pos}", 
                               help=f"Position {pos}"):
                        if 'human_move' not in st.session_state:
                            st.session_state.human_move = pos
            else:
                with col3:
                    if st.button(display_value if display_value else "⬜", 
                               key=f"cell_{pos}", 
                               help=f"Position {pos}"):
                        if 'human_move' not in st.session_state:
                            st.session_state.human_move = pos

def main():
    st.set_page_config(page_title="Tic-Tac-Toe RL Bot", page_icon="🤖", layout="wide")
    
    st.title("🤖 Tic-Tac-Toe Reinforcement Learning Bot")
    st.markdown("**Train an AI bot using epsilon-greedy Q-learning to master tic-tac-toe!**")
    
    # Initialize session state
    if 'bot' not in st.session_state:
        st.session_state.bot = TicTacToeBot()
    if 'game' not in st.session_state:
        st.session_state.game = TicTacToeGame()
    if 'game_mode' not in st.session_state:
        st.session_state.game_mode = 'training'
    if 'training_log' not in st.session_state:
        st.session_state.training_log = []
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Training parameters
        st.subheader("Training Parameters")
        num_games = st.slider("Number of Training Games", 100, 5000, 1000, 100)
        epsilon = st.slider("Epsilon (Exploration Rate)", 0.01, 0.5, 0.1, 0.01)
        learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
        
        # Update bot parameters
        st.session_state.bot.epsilon = epsilon
        st.session_state.bot.learning_rate = learning_rate
        
        # Training controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Train Bot"):
                with st.spinner("Training in progress..."):
                    progress_bar = st.progress(0)
                    log = train_bot(st.session_state.bot, num_games, progress_bar)
                    st.session_state.training_log.extend(log)
                    progress_bar.progress(1.0)
                st.success(f"Training completed! Played {num_games} games.")
        
        with col2:
            if st.button("🔄 Reset Bot"):
                st.session_state.bot = TicTacToeBot(epsilon, learning_rate)
                st.session_state.training_log = []
                st.success("Bot reset!")
        
        # Game mode selection
        st.subheader("Game Mode")
        if st.button("🎮 Play vs Bot"):
            st.session_state.game_mode = 'human'
            st.session_state.game = TicTacToeGame()
            st.success("Human vs Bot mode activated!")
        
        if st.button("🤖 Watch Bot vs Bot"):
            st.session_state.game_mode = 'bot_vs_bot'
            st.session_state.game = TicTacToeGame()
        
        # Save/Load bot
        st.subheader("Save/Load")
        if st.button("💾 Save Bot"):
            with open("tic_tac_toe_bot.pkl", "wb") as f:
                pickle.dump(st.session_state.bot, f)
            st.success("Bot saved!")
        
        if st.button("📁 Load Bot"):
            try:
                with open("tic_tac_toe_bot.pkl", "rb") as f:
                    st.session_state.bot = pickle.load(f)
                st.success("Bot loaded!")
            except FileNotFoundError:
                st.error("No saved bot found!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Game Board")
        
        if st.session_state.game_mode == 'human':
            # Human vs Bot game
            if not st.session_state.game.game_over:
                if st.session_state.game.current_player == 1:
                    st.info("Your turn! You are ❌")
                else:
                    st.info("Bot's turn... 🤖")
            
            display_board(st.session_state.game.board)
            
            # Handle human move
            if 'human_move' in st.session_state and st.session_state.game.current_player == 1:
                move = st.session_state.human_move
                if st.session_state.game.make_move(move, 1):
                    del st.session_state.human_move
                    st.rerun()
            
            # Handle bot move
            if st.session_state.game.current_player == 2 and not st.session_state.game.game_over:
                time.sleep(0.5)  # Add small delay for better UX
                bot_move = st.session_state.bot.choose_action(st.session_state.game.board, training=False)
                if bot_move != -1:
                    st.session_state.game.make_move(bot_move, 2)
                    st.rerun()
            
            # Game over message
            if st.session_state.game.game_over:
                if st.session_state.game.winner == 1:
                    st.success("🎉 You won!")
                elif st.session_state.game.winner == 2:
                    st.error("🤖 Bot won!")
                else:
                    st.warning("🤝 It's a draw!")
                
                if st.button("🔄 New Game"):
                    st.session_state.game = TicTacToeGame()
                    st.rerun()
        
        elif st.session_state.game_mode == 'bot_vs_bot':
            # Bot vs Bot demonstration
            st.info("Bot vs Bot game in progress...")
            display_board(st.session_state.game.board)
            
            if not st.session_state.game.game_over:
                if st.button("▶️ Next Move"):
                    current_player = st.session_state.game.current_player
                    bot_move = st.session_state.bot.choose_action(st.session_state.game.board, training=False)
                    if bot_move != -1:
                        st.session_state.game.make_move(bot_move, current_player)
                        st.rerun()
            else:
                if st.session_state.game.winner:
                    st.success(f"Player {st.session_state.game.winner} won!")
                else:
                    st.warning("It's a draw!")
                
                if st.button("🔄 New Game"):
                    st.session_state.game = TicTacToeGame()
                    st.rerun()
        
        else:
            # Training mode display
            st.info("🎯 Ready to train! Use the sidebar to configure and start training.")
            display_board(np.zeros((3, 3)))
    
    with col2:
        # Statistics
        st.subheader("📊 Bot Statistics")
        stats = st.session_state.bot.training_stats
        
        st.metric("Games Played", stats['games_played'])
        st.metric("Wins", stats['wins'])
        st.metric("Losses", stats['losses'])
        st.metric("Draws", stats['draws'])
        
        if stats['games_played'] > 0:
            win_rate = stats['wins'] / stats['games_played'] * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        st.metric("Q-Table Size", len(st.session_state.bot.q_table))
        
        # Training log
        if st.session_state.training_log:
            st.subheader("📝 Training Log")
            for log_entry in st.session_state.training_log[-10:]:  # Show last 10 entries
                st.text(log_entry)
        
        # Bot parameters display
        st.subheader("🎛️ Current Parameters")
        st.text(f"Epsilon: {st.session_state.bot.epsilon}")
        st.text(f"Learning Rate: {st.session_state.bot.learning_rate}")
        st.text(f"Discount Factor: {st.session_state.bot.discount_factor}")

if __name__ == "__main__":
    main()