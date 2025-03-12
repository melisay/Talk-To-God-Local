# TalkToGod-local

A Raspberry Pi-based conversational AI that uses OpenAI's ChatGPT and ElevenLabs TTS to simulate a witty, sarcastic version of God. This project listens for your speech, processes your requests, and responds with humorous audio.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- Speech recognition using the default microphone via the SpeechRecognition library and Google Speech API.
- Text-to-speech using the ElevenLabs API.
- Integration with ChatGPT for dynamic, conversational responses.
- Caching of responses to reduce latency and API calls.
- Idle mode and interruption handling.
- Modes for impressions, compliments, motivational quotes, voice switching, and more.

## Requirements

- **Hardware:**
  - Raspberry Pi
  - USB Microphone
  - Speakers or headphones

- **Software:**
  - Python 3.x
  - Git
  - `mpg123` for audio playback

- **Python Libraries:**
  - `openai`
  - `requests`
  - `speech_recognition`
  - `pyaudio`
  - `elevenlabs`
  - `python-dotenv`
  - `logging`
  - and others (see `requirements.txt` if available)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/melisay/TalkToGod-local.git
   cd TalkToGod-local

## Configuration

1. **Environment Variables::**
   ```bash
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   OPENAI_API_KEY=your_openai_api_key
   
2. **Voice Settings:**
- The project uses preset voices (VOICE_NIKKI) and (VOICE_TOM). 
- You can change these in the source code if needed.

3. **Paths:**
- Adjust paths such as BASE_DIR and CACHE_DIR in the source code if required

## Usage

1. **Running the Application::**
   ```bash
   python3 localgod.py

2. **Running the Application With Vosk::**
   ```bash
   python3 localgod-vosk.py

3. **Updating the Code:::**
   ```bash
   git pull

## Troubleshooting

No Valid Input Devices:
- Ensure your USB microphone is properly connected and set as the default input device.

API Errors:
- Verify that your .env file has the correct API keys and that your network connection is active.

Audio Playback Issues:
- Confirm that mpg123 is installed and functioning correctly.

## Contributing

If you'd like to contribute, please fork the repository and create a pull request with your changes. Follow the coding standards and guidelines outlined in the repository.

## Licence
MIT License