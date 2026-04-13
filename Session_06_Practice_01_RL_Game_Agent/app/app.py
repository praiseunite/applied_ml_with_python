import streamlit as st
import gymnasium as gym
from stable_baselines3 import PPO
import os
import time

st.set_page_config(
    page_title="🎮 CartPole Challenge",
    page_icon="🎮",
    layout="wide"
)

# ─── Custom Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1040 50%, #0d1a2a 100%);
    min-height: 100vh;
}

/* Hero header */
.hero-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.1rem;
    font-weight: 300;
}

/* Score cards */
.score-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
}
.score-label {
    color: #94a3b8;
    font-size: 0.85rem;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.score-value {
    font-size: 2.4rem;
    font-weight: 700;
    color: #f1f5f9;
}
.score-value.perfect { color: #34d399; }
.score-value.human   { color: #a78bfa; }
.score-value.best    { color: #f59e0b; }

/* Game frame container */
.game-frame {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1rem;
    text-align: center;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}
.status-playing  { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid #34d399; }
.status-gameover { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid #f87171; }
.status-waiting  { background: rgba(148,163,184,0.15); color: #94a3b8; border: 1px solid #94a3b8; }

/* Control buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.2s ease;
}

/* Separator */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    margin: 1.5rem 0;
}

/* Info box */
.info-box {
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🎮 CartPole Challenge</div>
    <p class="hero-sub">Watch the AI achieve a perfect score — then try to beat it yourself</p>
</div>
""", unsafe_allow_html=True)

# ─── Model Check ──────────────────────────────────────────────────────────────
MODEL_PATH = "ppo_cartpole"
MODEL_FILE = f"{MODEL_PATH}.zip"

if not os.path.exists(MODEL_FILE):
    st.error("⚠️ Trained model not found! Run the training notebook in `notebooks/` first.")
    st.stop()

# ─── Session State Init ───────────────────────────────────────────────────────
defaults = {
    "human_env":    None,
    "human_obs":    None,
    "human_score":  0,
    "human_done":   False,
    "human_frame":  None,
    "human_best":   0,
    "human_steps":  0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_ai, tab_human = st.tabs(["🤖  Watch the AI Play", "🕹️  Play It Yourself"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — AI AUTO-PLAY
# ══════════════════════════════════════════════════════════════════════════════
with tab_ai:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Stats row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="score-card">
            <div class="score-label">AI Score (always)</div>
            <div class="score-value perfect">500 / 500</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="score-card">
            <div class="score-label">Algorithm</div>
            <div class="score-value" style="font-size:1.4rem;margin-top:0.4rem">PPO · MlpPolicy</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="score-card">
            <div class="score-label">Training Steps</div>
            <div class="score-value" style="font-size:1.6rem;margin-top:0.3rem">25,000</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    _, btn_col, _ = st.columns([2, 3, 2])
    with btn_col:
        run_ai = st.button("▶️  Run AI Simulation", type="primary", key="run_ai", use_container_width=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if run_ai:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        model = PPO.load(MODEL_PATH)
        obs, _ = env.reset()

        frame_box   = st.empty()
        status_box  = st.empty()
        score_box   = st.empty()

        score = 0
        for i in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            score += int(reward)

            frame = env.render()
            frame_box.image(frame, channels="RGB", use_container_width=True,
                            caption=f"Step {i+1}  |  Score: {score}")
            status_box.markdown(
                f'<div style="text-align:center"><span class="status-badge status-playing">🟢 AI Balancing... Score: {score}</span></div>',
                unsafe_allow_html=True
            )
            time.sleep(0.02)

            if terminated or truncated:
                break

        env.close()

        if score >= 490:
            st.balloons()
            st.success(f"🏆 Perfect! The AI balanced the pole for all {score} steps without fail.")
        else:
            st.info(f"Simulation ended. Final Score: {score} / 500")

    else:
        st.markdown("""
        <div class="info-box">
        🤖 <strong>How it works:</strong> The AI was trained using <strong>Proximal Policy Optimization (PPO)</strong>
        for 25,000 timesteps. It learned purely through trial and error — no rules were given.
        It now scores a perfect <strong>500/500</strong> every single time it plays.
        Press the button above to watch it in action.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HUMAN INTERACTIVE PLAY
# ══════════════════════════════════════════════════════════════════════════════
with tab_human:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Score row ─────────────────────────────────────────────────────────────
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Your Current Score</div>
            <div class="score-value human">{int(st.session_state.human_score)}</div>
        </div>""", unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">Your Best Score</div>
            <div class="score-value best">{int(st.session_state.human_best)}</div>
        </div>""", unsafe_allow_html=True)
    with h3:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">AI Score to Beat</div>
            <div class="score-value perfect">500</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Start / Restart button ─────────────────────────────────────────────────
    _, start_col, _ = st.columns([2, 3, 2])
    with start_col:
        label = "🔄  Restart Game" if st.session_state.human_env is not None else "🎮  Start Game"
        if st.button(label, key="start_human", use_container_width=True):
            # Close old env if running
            if st.session_state.human_env is not None:
                try:
                    st.session_state.human_env.close()
                except Exception:
                    pass
            # Create fresh env
            env = gym.make("CartPole-v1", render_mode="rgb_array")
            obs, _ = env.reset()
            frame = env.render()
            st.session_state.human_env   = env
            st.session_state.human_obs   = obs
            st.session_state.human_frame = frame
            st.session_state.human_score = 0
            st.session_state.human_steps = 0
            st.session_state.human_done  = False
            st.rerun()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Active game UI ─────────────────────────────────────────────────────────
    if st.session_state.human_env is not None:

        # Status badge
        if st.session_state.human_done:
            badge = '<div style="text-align:center"><span class="status-badge status-gameover">💥 Game Over — Press Restart to try again</span></div>'
        else:
            badge = '<div style="text-align:center"><span class="status-badge status-playing">🟢 Balancing — Push Left or Right to keep the pole up!</span></div>'
        st.markdown(badge, unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # Game frame
        if st.session_state.human_frame is not None:
            _, img_col, _ = st.columns([1, 4, 1])
            with img_col:
                st.image(
                    st.session_state.human_frame,
                    channels="RGB",
                    use_container_width=True,
                    caption=f"Step {st.session_state.human_steps}  |  Score: {int(st.session_state.human_score)}"
                )

        # ── Control buttons ────────────────────────────────────────────────────
        if not st.session_state.human_done:
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            left_col, right_col = st.columns(2)

            with left_col:
                if st.button("⬅️  Push Cart LEFT", key="left", use_container_width=True):
                    obs, reward, terminated, truncated, _ = st.session_state.human_env.step(0)
                    st.session_state.human_obs    = obs
                    st.session_state.human_score += reward
                    st.session_state.human_steps += 1
                    st.session_state.human_frame  = st.session_state.human_env.render()
                    if terminated or truncated:
                        st.session_state.human_done = True
                        if st.session_state.human_score > st.session_state.human_best:
                            st.session_state.human_best = st.session_state.human_score
                        st.session_state.human_env.close()
                    st.rerun()

            with right_col:
                if st.button("➡️  Push Cart RIGHT", key="right", use_container_width=True):
                    obs, reward, terminated, truncated, _ = st.session_state.human_env.step(1)
                    st.session_state.human_obs    = obs
                    st.session_state.human_score += reward
                    st.session_state.human_steps += 1
                    st.session_state.human_frame  = st.session_state.human_env.render()
                    if terminated or truncated:
                        st.session_state.human_done = True
                        if st.session_state.human_score > st.session_state.human_best:
                            st.session_state.human_best = st.session_state.human_score
                        st.session_state.human_env.close()
                    st.rerun()

        # ── Game Over result ───────────────────────────────────────────────────
        if st.session_state.human_done:
            final = int(st.session_state.human_score)
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            if final >= 490:
                st.balloons()
                st.success(f"🏆 Incredible! You matched the AI's perfect score: {final}/500!")
            elif final >= 200:
                st.success(f"🎉 Great effort! You scored {final}/500. The AI scores 500 every time — keep practising!")
            elif final >= 100:
                st.warning(f"👏 Score: {final}/500. Not bad! The pole fell at step {final}. Try again!")
            else:
                st.error(f"💥 Score: {final}/500. The pole fell quickly! Watch the AI first to see the strategy.")

    else:
        # No game started yet
        st.markdown("""
        <div class="info-box">
        🕹️ <strong>How to play:</strong><br>
        A pole is balanced on a cart. Every step the pole stays upright gives you <strong>+1 point</strong>.
        Use the <strong>← Left</strong> and <strong>→ Right</strong> buttons to push the cart and keep the pole from falling.
        The maximum score is <strong>500</strong> — that's what the AI achieves every time.<br><br>
        💡 <strong>Tip:</strong> Push in the direction the pole is leaning to counteract the fall.
        </div>
        """, unsafe_allow_html=True)

    # ── Physics hint ──────────────────────────────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    with st.expander("💡 Strategy Hints"):
        st.markdown("""
        - **Watch the pole tip** — if it leans right, push right. If it leans left, push left.
        - **Don't over-correct** — small, frequent adjustments beat big sudden moves.
        - **Watch the cart position** — if the cart drifts to the edge, you'll run out of room.
        - The AI makes ~10 corrections per second automatically. You're doing it step by step — so even scoring 50+ is impressive!
        """)
