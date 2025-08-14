# 🤖 Tic-Tac-Toe Reinforcement Learning Bot

A sophisticated AI bot that learns to play Tic-Tac-Toe using Q-Learning and epsilon-greedy strategy. Built with Python and Streamlit for an interactive web interface.

## 🎯 Features

- **Q-Learning Algorithm**: Bot learns optimal strategies through self-play
- **Epsilon Decay**: Starts with high exploration (90%) and gradually becomes more strategic (1%)
- **Interactive UI**: Clean Streamlit interface for training and playing
- **Massive Training**: Supports up to 500,000 training games
- **Save/Load**: Persistent bot storage in JSON format
- **Real-time Stats**: Track win rates, Q-table size, and training progress
- **Multiple Game Modes**: Human vs Bot, Bot vs Bot demonstration

## 🚀 How It Works

1. **Training Phase**: Bot plays thousands of games against itself
2. **Epsilon-Greedy**: Balances exploration (random moves) vs exploitation (best known moves)
3. **Q-Learning Updates**: Learns from wins (+1), losses (-1), and draws (0)
4. **Epsilon Decay**: Reduces exploration over time (0.9 → 0.01)
5. **Human Play**: Test your skills against the trained AI

## 📋 Requirements

- Python 3.12+
- Streamlit
- NumPy

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/tic-tac-toe-rl-bot.git
   cd tic-tac-toe-rl-bot
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run tictactoe.py
   ```

## 🎮 Usage

### Training the Bot
1. Set training parameters in the sidebar:
   - **Games**: 1,000 - 500,000 (start with 10,000)
   - **Initial Epsilon**: 0.9 (90% exploration)
   - **Epsilon Decay**: 0.01 per game
   - **Learning Rate**: 0.1

2. Click "🎯 Train Bot" and watch the progress
3. Monitor win rates and epsilon decay in real-time

### Playing Against the Bot
1. After training, click "🎮 Play vs Bot"
2. You are ❌, bot is ⭕
3. Click empty squares to make moves
4. Try to beat the AI!

### Saving Your Bot
- Click "💾 Save Bot" to preserve trained model
- Bot saves as JSON file for easy loading later
- Click "📁 Load Bot" to restore saved bot

## 📊 Training Recommendations

- **Beginner Bot**: 1,000 games (~1 minute)
- **Intermediate Bot**: 10,000 games (~5 minutes)  
- **Advanced Bot**: 50,000+ games (~20+ minutes)
- **Master Bot**: 200,000+ games (very strong play)

## 🧠 Algorithm Details

### Q-Learning Formula
```
Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
```

Where:
- `α` = Learning rate (0.1)
- `r` = Reward (+1 win, -1 loss, 0 draw)
- `γ` = Discount factor (0.9)

### Epsilon-Greedy Strategy
- **High Epsilon** (0.9): Mostly random moves (exploration)
- **Low Epsilon** (0.01): Mostly optimal moves (exploitation)
- **Decay**: Reduces by 0.01 each game until minimum reached

## 🔧 Project Structure

```
tic-tac-toe-rl-bot/
├── tictactoe.py           # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore           # Git ignore rules
└── tic_tac_toe_bot.json # Saved bot (created after training)
```

## 📈 Performance

After proper training, the bot typically achieves:
- **95%+ win rate** against random play
- **Near-optimal play** in most positions
- **Strategic blocking** of opponent wins
- **Tactical setup** of winning combinations

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🎯 Future Enhancements

- [ ] Neural network implementation
- [ ] Different board sizes (4x4, 5x5)
- [ ] Tournament mode
- [ ] Bot vs Bot competitions
- [ ] Advanced visualization
- [ ] Mobile-responsive design

## 👨‍💻 Author

**Sumant** - *Initial work*

## 🙏 Acknowledgments

- Reinforcement Learning concepts from Sutton & Barto
- Streamlit for the amazing web framework
- NumPy for efficient array operations