# Integrated LeetCode Interview System

This system provides a fully voice-interactive technical interview experience, combining an AI interviewer with a LeetCode coding assistant.

## Setup

1. Install the required dependencies:
```bash
pip install google-genai opencv-python pyaudio pillow mss pyautogui
```

2. Set up your environment:
- Get your API key from Google AI Studio (https://makersuite.google.com/app/apikey)
- Set the environment variable:
  ```bash
  export GOOGLE_API_KEY='your-api-key-here'
  ```
- Have the LeetCode problem page ready to switch to when prompted
- Connect a microphone and speakers/headphones (headphones recommended)

## Usage

1. Run the integrated interview:
```bash
python integrated_interview.py
```

2. The system will:
   - Start a voice-based interview
   - Ask for your name and confirm readiness
   - Present the coding problem and handle any clarifying questions
   - When you're ready to code, prompt you to switch to the LeetCode interface
   - Transition to the coding assistant phase which will help guide your implementation

## Interview Flow

1. Voice Interview Phase:
   - Speak with the AI interviewer naturally
   - The system will transcribe your voice and respond with audio
   - Ask questions about the problem and get clarification
   - Indicate when you're ready to start coding

2. LeetCode Assistant Phase:
   - Continue voice interaction while coding
   - Share your screen showing the LeetCode environment
   - Get real-time guidance and hints through voice
   - Receive code analysis and optimization suggestions

## Important Notes

- Use headphones to prevent audio feedback
- Speak clearly and wait for the interviewer to finish before responding
- Have the LeetCode environment ready to switch to when prompted
- The system will automatically transition between phases based on your readiness
- You can ask for hints or clarification at any time during either phase

## Troubleshooting

If you experience audio issues:
- Ensure your microphone and speakers are properly connected
- Check that they are set as your system's default devices
- Verify that no other applications are using the audio devices
- Try restarting the application if audio becomes desynchronized
