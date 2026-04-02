# Practice 01B (Bonus): Checkers (Drafts) RL Engine

Welcome to the Bonus Practice Session! 

In the primary session (`Practice_01_RL_Game_Agent`), we taught an AI to balance a pole using physical mechanics. Here, we tackle a classic board game: **Checkers (Drafts)**. 

This bonus module focuses heavily on **Reward Engineering**. You will clearly see how mapping points additions (captures) and points deductions (losing a piece) train the neural network to develop a winning strategy.

---

## 🎯 Learning Objectives

By the end of this session, you will be able to:
1. Implement a Multi-Agent environment using `pettingzoo`.
2. Visualize explicitly how mathematical reward vectors act as the "teacher" for an AI.
3. Incorporate **Sound Effects** into a machine learning application frontend to enhance the user experience.
4. Understand the time-complexity differences between single-agent physical mechanics (CartPole) vs multi-agent board games (Checkers).

---

## 📖 Part 1: See the Additions & Deductions in Action

Unlike CartPole where you get +1 for merely surviving, Checkers requires a very strict Reward Function:
- **+1 Point**: For regular forward movement.
- **+10 Points**: For capturing an opponent's piece! (Positive Addition)
- **-10 Points**: If your piece gets captured! (Negative Deduction)
- **+100 Points**: For winning the entire game!

In the `notebooks/` directory, open `Drafts_Checkers_Lab.ipynb`. This notebook installs `pettingzoo`, prints out the mathematical reward structure, and begins training our deep learning agent using PPO (Policy Optimization).

---

## 📖 Part 2: Connect the AI to a Web Interface with Sound!

Once you have generated your `checkers_agent.zip` brain:
1. Navigate to the `app/` directory.
2. Run the deployment environment:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Watch the frontend visualize the pieces moving on the digital board. When the AI makes a capture, the UI triggers a `capture.wav` sound! When a player wins, it triggers a `win.wav` sound!

---

## 🚀 Folder Contents

1. **`README.md`** -> This documentation.
2. **`notebooks/Drafts_Checkers_Lab.ipynb`** -> Training lab showing the mathematical reward structure mapping.
3. **`app/app.py`** -> Streamlit interface featuring the interactive game board and sound playback mechanisms.
4. **`app/sounds/`** -> Pre-generated sound effects for `capture` and `win`.

---
*© 2024 Aptech Limited — For Educational Use*
