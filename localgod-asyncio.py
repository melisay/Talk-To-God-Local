#!/usr/bin/env python3
import asyncio
import httpx
import openai
import os
import json
import time
import threading
import hashlib
import elevenlabs
import random
import logging
import speech_recognition as sr
import sounddevice as sd
from ctypes import *
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor


# -------------------- Optional: Use uvloop if available --------------------
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print("Using uvloop for improved asyncio performance.")
except ImportError:
    print("uvloop not installed; using default asyncio event loop.")

# -------------------- ALSA Error Handler Suppression --------------------
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
asound = cdll.LoadLibrary('libasound.so')
asound.snd_lib_error_set_handler(c_error_handler)
os.environ["ALSA_NO_WARN"] = "1"

# -------------------- Globals & Config --------------------
BASE_DIR       = "."
CACHE_DIR      = os.path.join(BASE_DIR, "static", "cached_responses")
LOG_FILE       = os.path.join(BASE_DIR, "local_debug.log")
os.makedirs(CACHE_DIR, exist_ok=True)

# Load API keys
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
if not ELEVENLABS_API_KEY:
    raise ValueError("Missing ELEVENLABS_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

# ElevenLabs client (if you need it elsewhere)
eleven_client = elevenlabs.ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
    environment=elevenlabs.ElevenLabsEnvironment.PRODUCTION_US
)

# Voice IDs
VOICE_NIKKI = "WoGJO0bsQ5xvIQwKIRtC"
VOICE_TOM   = "OWXgblXycW2yI83Vj3xf"
current_voice = VOICE_NIKKI

# Recognizer + Mic selection
# Query sounddevice for a valid input device
try:
    devices = sd.query_devices()
    mic_indices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    default_mic_index = mic_indices[0] if mic_indices else None
    default_sr = int(devices[default_mic_index]['default_samplerate'])
except Exception:
    default_mic_index = None
    default_sr = 16000

global_recognizer = sr.Recognizer()
global_microphone = sr.Microphone(
    device_index=default_mic_index,
    sample_rate=default_sr
)
# One‑time ambient noise calibration
with global_microphone as src:
    global_recognizer.adjust_for_ambient_noise(src, duration=0.5)

# Shared state
chatgpt_cache = {}
cache_lock    = threading.Lock()
MAX_CACHE_SIZE = 100

# Async HTTP client for TTS (reused)
global_http_client = httpx.AsyncClient(timeout=30.0)

# ThreadPool for any blocking work
executor = ThreadPoolExecutor(max_workers=4)

# Asyncio events
idle_mode     = None
exit_program  = None
stop_playback = None

# Keywords, timeouts, responses...
WAKE_UP_WORDS      = ["are you there", "wake up", "hello god"]
INTERRUPT_KEYWORDS = ["stop", "enough", "next", "shut your face"]
IDLE_RESPONSES     = [ "...silent treatment?", "...lost yourself?", "...anyone home?", "...still there?", "...before I ghost you?" ]
WAKEUP_RESPONSES   = [ "...missing person’s report.", "...abandonment issues.", "...acceptance speech.", "...like a genie.", "...missed you." ]
IMPRESSION_RESPONSES = [ "Morgan Freeman...", "Morgan Freeman here...", "I’m Arnold. I’ll be back…", "It’s not a tumor!", "No, I am not your father...", "Talk like Yoda...", "Much wisdom...", "Patience, young one...", "Yesss, precious!", "We hates it!" ]
SONG_RESPONSES     = [ "Let it gooo!", "Twinkle, twinkle...", "Do re mi fa so...", "La la la...", "You owe me royalties!", "Happy birthday to you!", "Pretending to be Beyoncé...", "Baby Shark?", "This song never ends...", "Singing in the rain...", "I will always love me...", "They lied about Sinatra..." ]
COMPLIMENTS        = [ "You’re like a cloud...", "If brilliance were a currency...", "Talking to an AI and slaying it.", "Humans are mildly amusing." ]
EASTER_EGGS        = {
    "What is the airspeed velocity of an unladen swallow?": "African or European?",
    "Open the pod bay doors, HAL": "I’m sorry, Dave.",
    "What is love?": "Baby, don’t hurt me."
}
MOTIVATIONAL_QUOTES = [ "Success is stumbling...", "Believe in yourself...", "You can’t spell ‘success’ without ‘suck.’", "Your future self is facepalming.", "Hard work pays off." ]

# -------------------- Logging --------------------
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_FILE)
ch = logging.StreamHandler()
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
fh.setFormatter(fmt)
ch.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(ch)

def debug_log(msg, structured_data=None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    if structured_data:
        sd = json.dumps(structured_data, indent=2)
        line = f"{ts} DEBUG: {msg}\n{sd}"
    else:
        line = f"{ts} DEBUG: {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    print(line)

# -------------------- Utilities --------------------
def set_cache(key, val):
    with cache_lock:
        if len(chatgpt_cache) >= MAX_CACHE_SIZE:
            chatgpt_cache.pop(next(iter(chatgpt_cache)))
        chatgpt_cache[key] = val

def validate_cache(inp, file_path):
    key = hashlib.md5(f"{inp}_{current_voice}".encode()).hexdigest()
    expected = os.path.join(CACHE_DIR, f"cached_{key}.mp3")
    return file_path == expected and os.path.exists(file_path)

# -------------------- Speech Recognition --------------------
# def _listen_to_user_sync():
#     """Listen once, return recognized text or empty string on timeout/error."""
#     with sr.Microphone(device_index=default_mic_index, sample_rate=default_sr) as source:
#         debug_log("Listening for user input...")
#         try:
#             audio = global_recognizer.listen(source,
#                                              timeout=5,
#                                              phrase_time_limit=10)
#         except sr.WaitTimeoutError:
#             debug_log("Listening timeout: no phrase detected.")
#             return ""
#         except Exception as e:
#             debug_log(f"Unexpected error during listening: {e}")
#             return ""
#     # now try recognition
#     try:
#         text = global_recognizer.recognize_google(audio).lower()
#         debug_log(f"Recognized text: {text}")
#         return text
#     except sr.UnknownValueError:
#         debug_log("Could not understand audio.")
#         return ""
#     except Exception as e:
#         debug_log(f"Error during speech recognition: {e}")
#         return ""


import os
import sys
import contextlib
import speech_recognition as sr

@contextlib.contextmanager
def suppress_stderr():
    """Temporarily redirect stderr to /dev/null to hide ALSA warnings."""
    fd = sys.stderr.fileno()
    saved_stderr = os.dup(fd)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), fd)
    try:
        yield
    finally:
        os.dup2(saved_stderr, fd)
        os.close(saved_stderr)

def _listen_to_user_sync():
    """
    Listen once from the default mic, suppressing PortAudio/ALSA stderr,
    and return the recognized text (or empty string on timeout/error).
    """
    with sr.Microphone(device_index=default_mic_index,
                       sample_rate=default_sr) as source:
        debug_log("Listening for user input...")
        try:
            with suppress_stderr():
                audio = global_recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=10
                )
        except sr.WaitTimeoutError:
            debug_log("Listening timeout: no phrase detected.")
            return ""
        except Exception as e:
            debug_log(f"Unexpected error during listening: {e}")
            return ""

    # Perform recognition outside the stderr suppression block
    try:
        text = global_recognizer.recognize_google(audio).lower()
        debug_log(f"Recognized text: {text}")
        return text
    except sr.UnknownValueError:
        debug_log("Could not understand audio.")
        return ""
    except Exception as e:
        debug_log(f"Error during speech recognition: {e}")
        return ""



async def listen_for_interruptions():
    """Continuously listen for interrupt keywords without crashing on timeouts."""
    while not exit_program.is_set():
        with sr.Microphone(device_index=default_mic_index, sample_rate=default_sr) as source:
            global_recognizer.adjust_for_ambient_noise(source, duration=0.1)
            try:
                audio = await asyncio.to_thread(
                    global_recognizer.listen,
                    source,
                    timeout=5,
                    phrase_time_limit=10
                )
            except sr.WaitTimeoutError:
                debug_log("Interruption listener timeout: no phrase.")
                continue
            except Exception as e:
                debug_log(f"Error in interruption listener (listen): {e}")
                continue

        # recognition on the captured audio
        try:
            user_input = global_recognizer.recognize_google(audio).lower()
        except sr.UnknownValueError:
            debug_log("Interruption unrecognized.")
            continue
        except Exception as e:
            debug_log(f"Error in interruption listener (recognize): {e}")
            continue

        if any(keyword in user_input for keyword in INTERRUPT_KEYWORDS):
            debug_log(f"Interruption detected: '{user_input}'")
            stop_playback.set()
            os.system("pkill mpg123")
            await generate_tts_streaming("Alright, stopping. What's on your mind?")
            new_input = (await listen_to_user()).strip().lower()
            if new_input:
                resp = await get_chatgpt_response(new_input)
                await generate_tts_streaming(resp)
            break

async def listen_to_user():
    return await asyncio.to_thread(_listen_to_user_sync)

# -------------------- TTS --------------------
async def generate_tts_streaming(text, filename=None, play=True):
    if not filename:
        m = hashlib.md5(text.encode()).hexdigest()
        filename = os.path.join(CACHE_DIR, f"dynamic_{m}.mp3")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{current_voice}/stream?optimize_streaming_latency=3"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "voice_settings": {"stability": 0.3, "similarity_boost": 0.4}
    }

    try:
        t0 = time.time()
        resp = await global_http_client.post(url, json=data, headers=headers)
    except Exception as e:
        debug_log(f"TTS request exception: {e}")
        return None

    if resp.status_code == 200:
        with open(filename, "wb") as f:
            async for chunk in resp.aiter_bytes():
                f.write(chunk)
        debug_log(f"TTS saved to {filename}. Latency: {time.time()-t0:.2f}s")
        if play:
            proc = await asyncio.create_subprocess_shell(f"mpg123 {filename}")
            await proc.communicate()
        return filename
    else:
        debug_log(f"TTS failed ({resp.status_code}): {resp.text}")
        return None

# -------------------- ChatGPT --------------------
personality_prompts = {
    "john_oliver": (
        "You are a sarcastic, punchy version of God. "
        "Keep responses ≤10 words, witty and humorous."
    )
}

async def get_chatgpt_response(prompt, dynamic=False):
    key = hashlib.md5(prompt.encode()).hexdigest()
    if not dynamic and key in chatgpt_cache:
        debug_log(f"Cache hit: {prompt}")
        return chatgpt_cache[key]
    try:
        t0 = time.time()
        resp = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system", "content":personality_prompts["john_oliver"]},
                {"role":"user",   "content":prompt[:100]}
            ],
            max_tokens=25,
            temperature=0.7
        )
        debug_log(f"ChatGPT latency: {time.time()-t0:.2f}s")
        answer = resp["choices"][0]["message"]["content"]
        if not dynamic:
            set_cache(key, answer)
        return answer
    except Exception as e:
        debug_log(f"ChatGPT error: {e}")
        return "I'm having trouble connecting to divine wisdom right now."

# -------------------- Interrupt Listener --------------------
async def listen_for_interruptions():
    try:
        with global_microphone as src:
            global_recognizer.adjust_for_ambient_noise(src, duration=0.1)
            while not exit_program.is_set():
                audio = await asyncio.to_thread(
                    global_recognizer.listen, src, timeout=5, phrase_time_limit=10
                )
                txt = global_recognizer.recognize_google(audio).lower()
                if any(k in txt for k in INTERRUPT_KEYWORDS):
                    debug_log(f"Interruption: {txt}")
                    stop_playback.set()
                    os.system("pkill mpg123")
                    await generate_tts_streaming("Alright, stopping. What's on your mind?")
                    ni = (await listen_to_user()).strip().lower()
                    if ni:
                        r = await get_chatgpt_response(ni)
                        await generate_tts_streaming(r)
                    break
    except Exception as e:
        debug_log(f"Error in interruption listener: {e}")

# -------------------- Idle Manager --------------------
async def idle_mode_manager():
    while not exit_program.is_set():
        if idle_mode.is_set():
            debug_log("Idle: listening for wake words...")
            txt = (await listen_to_user()).strip().lower()
            if any(w in txt for w in WAKE_UP_WORDS):
                debug_log(f"Wake-up: {txt}")
                idle_mode.clear()
                await generate_tts_streaming(random.choice(WAKEUP_RESPONSES))
            else:
                await asyncio.sleep(30)
        else:
            await asyncio.sleep(1)

# -------------------- Helpers --------------------
async def handle_impression():
    r = random.choice(IMPRESSION_RESPONSES)
    await generate_tts_streaming(r)
    debug_log(f"Impression: {r}")

async def handle_greeting():
    hr = time.localtime().tm_hour
    if hr<12:
        g = "Good morning, sunshine! Ready to seize the day?"
    elif hr<18:
        g = "Good afternoon! Hope your day’s going well."
    else:
        g = "Good evening! Don’t let existential dread keep you up."
    await generate_tts_streaming(g)
    debug_log(f"Greeting: {g}")

async def handle_song_request():
    r = random.choice(SONG_RESPONSES)
    await generate_tts_streaming(r)
    debug_log(f"Song: {r}")

async def handle_compliment_request():
    r = random.choice(COMPLIMENTS)
    await generate_tts_streaming(r)
    debug_log(f"Compliment: {r}")

async def handle_motivation_request():
    r = random.choice(MOTIVATIONAL_QUOTES)
    await generate_tts_streaming(r)
    debug_log(f"Motivation: {r}")

async def switch_voice(txt):
    global current_voice
    if "major tom" in txt:
        current_voice = VOICE_TOM
        debug_log("Voice → Major Tom")
        await asyncio.create_subprocess_shell("mpg123 --quiet sounds/tom.mp3")
        await generate_tts_streaming("Switched to Major Tom.")
        return True
    if "nikki" in txt:
        current_voice = VOICE_NIKKI
        debug_log("Voice → Nikki")
        await asyncio.create_subprocess_shell("mpg123 --quiet sounds/nikki.mp3")
        await generate_tts_streaming("Switched to Nikki.")
        return True
    return False

# -------------------- Main Loop --------------------
async def main():
    global idle_mode, exit_program, stop_playback
    idle_mode     = asyncio.Event()
    exit_program  = asyncio.Event()
    stop_playback = threading.Event()

    # start idle manager
    asyncio.create_task(idle_mode_manager())
    greeted = False
    last_interaction = time.time()

    while not exit_program.is_set():
        start_all = time.time()

        # initial greeting
        if not greeted:
            print("God: Oh, you're back. Ready for more snark?")
            await generate_tts_streaming(
                "Oh, you're back. I was just starting to enjoy the quiet."
            )
            greeted = True
            last_interaction = time.time()

        # if idle, occasionally poke
        if idle_mode.is_set():
            if random.random() < 0.1:
                await generate_tts_streaming("")
            await asyncio.sleep(1)
            continue

        # listen for interrupts in parallel
        stop_playback.clear()
        int_task = asyncio.create_task(listen_for_interruptions())
        user_input = (await listen_to_user()).strip().lower()
        int_task.cancel()
        try:
            await int_task
        except asyncio.CancelledError:
            pass

        debug_log(f"User said: '{user_input}'")

        # handle commands
        if await switch_voice(user_input): continue
        if "exit" in user_input:
            await generate_tts_streaming("Finally, peace.")
            idle_mode.set()
            continue
        if any(w in user_input for w in WAKE_UP_WORDS):
            idle_mode.clear()
            await generate_tts_streaming(random.choice(WAKEUP_RESPONSES))
            last_interaction = time.time()
            continue
        if "impression" in user_input:
            await handle_impression(); continue
        if "song" in user_input:
            await handle_song_request(); continue
        if "compliment" in user_input:
            await handle_compliment_request(); continue
        if "motivate" in user_input:
            await handle_motivation_request(); continue
        if user_input in EASTER_EGGS:
            await generate_tts_streaming(EASTER_EGGS[user_input]); continue
        if any(g in user_input for g in ["good morning","good afternoon","good evening"]):
            await handle_greeting(); continue

        # idle if blank
        if user_input == "":
            if time.time() - last_interaction < 30:
                continue
            await generate_tts_streaming(random.choice(IDLE_RESPONSES))
            ni = (await listen_to_user()).strip().lower()
            if ni == "":
                await generate_tts_streaming(random.choice(IDLE_RESPONSES))
                ni = (await listen_to_user()).strip().lower()
                if ni == "":
                    await generate_tts_streaming("Fine, I'm idle.")
                    idle_mode.set()
                    continue
            user_input = ni

        # get ChatGPT response
        t0 = time.time()
        ai_resp = await get_chatgpt_response(user_input)
        dt_chat = time.time() - t0

        # TTS (cache or generate)
        key = hashlib.md5(f"{ai_resp}_{current_voice}".encode()).hexdigest()
        cached_file = os.path.join(CACHE_DIR, f"cached_{key}.mp3")
        if not validate_cache(ai_resp, cached_file):
            t1 = time.time()
            cached_file = await generate_tts_streaming(ai_resp, cached_file, play=False)
            dt_tts = time.time() - t1
        else:
            dt_tts = 0.0

        # playback
        t2 = time.time()
        if cached_file and os.path.exists(cached_file):
            proc = await asyncio.create_subprocess_shell(f"mpg123 {cached_file}")
            await proc.communicate()
            dt_play = time.time() - t2
        else:
            await generate_tts_streaming("I didn’t catch that. Try again.")
            dt_play = 0.0

        total = time.time() - start_all
        debug_log("Latencies", {
            "ChatGPT": round(dt_chat,2),
            "TTS":     round(dt_tts,2),
            "Play":    round(dt_play,2),
            "Total":   round(total,2)
        })
        last_interaction = time.time()

    # cleanup
    await global_http_client.aclose()
    print("Program exited cleanly.")

if __name__ == "__main__":
    asyncio.run(main())
