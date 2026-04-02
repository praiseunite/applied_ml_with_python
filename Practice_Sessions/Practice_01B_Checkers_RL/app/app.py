import streamlit as st
import base64
import time
import os

st.set_page_config(page_title="AI Checkers", page_icon="🔴")
st.title("🔴 AI Checkers: Reward Audio Visualization")
st.markdown("Watch the Game Engine execute! Because we mapped the Checkers AI logic to mathematical rewards, our UI can 'listen' to the math: Every time an agent receives **+10 points**, it plays a Capture Sound. Winning plays a Victory Sound!")

def play_sound(file_name):
    """Embeds an invisible HTML audio block that auto-plays a base64 encoded sound file."""
    path = os.path.join("sounds", file_name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f'<audio autoplay="true" src="data:audio/wav;base64,{b64}"></audio>'
            st.markdown(md, unsafe_allow_html=True)
    else:
        st.warning(f"Audio file '{file_name}' not found. Did you run `sound_generator.py`?")

if st.button("▶️ Start Physics/AI Checkers Sandbox", type="primary"):
    # We load the PettingZoo Multi-Agent RL Checker environment
    from pettingzoo.classic import checkers_v3
    
    frame_placeholder = st.empty()
    status_text = st.empty()
    
    st.info("Initializing multi-agent PettingZoo environment...")
    
    env = checkers_v3.env(render_mode="rgb_array")
    env.reset()
    
    # We iterate through the agents playing each other sequentially
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        # UI EVENT LISTENER (Triggered by underlying ML Math)
        if reward >= 10 and not termination:
             # Captured a piece!
             play_sound("capture.wav")
             st.toast(f"Boom! {agent} Captured a Piece! (+10 pts)", icon="🔥")
        elif termination and reward > 0:
             # Won the game!
             play_sound("win.wav")
             st.balloons()
             st.success(f"🏆 {agent} Won the Match! (+100 pts)")
             
        if termination or truncation:
            action = None
        else:
            # We sample random actions purely to demonstrate the engine visualization mechanics 
            # (Training a real checkers agent takes extreme computing)
            action = env.action_space(agent).sample()
            
        env.step(action)
        
        # Render the board to screen Native to Streamlit!
        frame = env.render()
        if frame is not None:
             frame_placeholder.image(frame, channels="RGB", caption=f"Turn: {agent}")
             
        status_text.text(f"Currently deciding: {agent} | Last Reward Sent: {reward}")
        
        time.sleep(0.5) # Slight pause to let humans see the move and hear the audio!
        
        if all(env.terminations.values()):
             break
             
    env.close()
