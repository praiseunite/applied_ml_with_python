import wave
import struct
import math
import os

os.makedirs("sounds", exist_ok=True)

# Generate a fast "Bloop" sound for capture
capture_file = "sounds/capture.wav"
sample_rate = 44100.0
capture_length = int(sample_rate * 0.15) # 150ms

with wave.open(capture_file, 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    for i in range(capture_length):
        freq = 600.0 + (i / capture_length) * 400.0 # Frequency slide up
        value = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
        data = struct.pack('<h', value)
        f.writeframesraw(data)

# Generate a "Ta-Da" sound for winning
win_file = "sounds/win.wav"
win_length = int(sample_rate * 0.8)

with wave.open(win_file, 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    for i in range(win_length):
        if i < win_length / 2:
            freq = 440.0 # A4
        else:
            freq = 554.37 # C#5 (Major Third, sounds happy)
            
        value = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * freq * (i / sample_rate)))
        data = struct.pack('<h', value)
        f.writeframesraw(data)

print("Sound files 'capture.wav' and 'win.wav' generated successfully in sounds/ directory.")
