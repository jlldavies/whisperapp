"""Run once to generate tests/fixtures/test_5sec.wav"""
import wave, struct, math, os
SAMPLE_RATE = 16000
DURATION = 5
filename = os.path.join(os.path.dirname(__file__), "test_5sec.wav")
with wave.open(filename, "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(SAMPLE_RATE)
    for i in range(SAMPLE_RATE * DURATION):
        val = int(32767 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE))
        f.writeframes(struct.pack("<h", val))
print(f"Written {filename}")
