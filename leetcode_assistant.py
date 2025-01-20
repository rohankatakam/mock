# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

To install the dependencies for this script, run:

``` 
pip install google-genai opencv-python pyaudio pillow mss pyautogui
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones. 

## Run

To run the LeetCode assistant:

```
python leetcode_assistant.py
```

The assistant will:
1. Capture your screen to see the LeetCode problem
2. Listen to your voice questions/comments
3. Provide verbal guidance and hints
"""

import asyncio
import base64
import io
import os
import sys
import traceback
import tkinter as tk
from tkinter import ttk, scrolledtext
import ttkthemes
import re
import time
import multiprocessing

import cv2
import pyaudio
import PIL.Image
import mss
import pyautogui  # For automated typing

import argparse

from google import genai

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.0-flash-exp"

DEFAULT_MODE = "screen"  # Always use screen mode for LeetCode

# Configure the model to be a LeetCode tutor
TUTOR_PROMPT = """You are an expert programming tutor specializing in LeetCode problems.
When you see a LeetCode problem on the screen:
1. First identify and explain the problem
2. Guide the user through the solution approach:
   - Help understand the problem constraints
   - Discuss possible solutions and their trade-offs
   - Explain time and space complexity
3. If the user is stuck, provide hints rather than complete solutions
4. When asked about specific parts of the code, explain the concepts in detail
5. Provide optimization suggestions when relevant

Keep responses concise but informative. Use clear technical language."""

CODE_ANALYZER_PROMPT = """You are an expert code analyzer specializing in detecting logical errors and improving code quality.
When analyzing code:
1. Identify potential logical errors and edge cases
2. Check for:
   - Off-by-one errors
   - Boundary conditions
   - Null/None handling
   - Integer overflow
   - Incorrect loop termination
3. Analyze time and space complexity
4. Suggest optimizations for:
   - Algorithm efficiency
   - Memory usage
   - Code readability
5. Compare with common patterns for this type of problem
6. When asked to provide a solution, format it as: 
   ```python
   [Your solution code here]
   ```

Be precise and thorough in your analysis. Focus on correctness first, then optimization."""

client = genai.Client(http_options={"api_version": "v1alpha"})

CONFIG = {
    "generation_config": {
        "response_modalities": ["AUDIO"],
    },
    "safety_settings": [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ],
}

pya = pyaudio.PyAudio()


def run_leetcode_gui():
    """Run the LeetCode GUI using the existing LeetCodeSolverGUI class"""
    try:
        # Import the GUI class from the existing module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(module_dir)
        from leetcode_solver_gui import LeetCodeSolverGUI, main
        
        # Use the main function from leetcode_solver_gui.py
        main()
    except Exception as e:
        print(f"Error in GUI: {e}")


class LeetCodeAssistant:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.audio_in_queue = asyncio.Queue()
        self.audio_out_queue = asyncio.Queue()
        self.session = None
        self.audio_stream = None
        self.analyzer_session = None
        self.last_screen = None
        self.code_detected = False
        self.editor_position = None
        self.gui_showing = False
        
        # Set up PyAutoGUI safety settings
        pyautogui.PAUSE = 0.5  # Add a 0.5 second pause between actions

    async def find_code_editor(self):
        """Locate the LeetCode code editor area"""
        try:
            # Look for the code editor region
            # You might need to adjust these coordinates based on your screen
            editor = pyautogui.locateOnScreen('leetcode_editor.png', confidence=0.8)
            if editor:
                self.editor_position = editor
                return True
        except pyautogui.ImageNotFoundException:
            return False
        return False

    async def type_solution(self, code):
        """Type the solution into the LeetCode editor"""
        if not self.editor_position:
            if not await self.find_code_editor():
                print("Could not locate code editor. Please position cursor manually.")
                return False

        try:
            # Extract code from markdown-style code block
            code_match = re.search(r'```python\n(.*?)```', code, re.DOTALL)
            if not code_match:
                print("No properly formatted code solution found")
                return False
                
            solution_code = code_match.group(1).strip()
            
            # Click in the editor area
            editor_center = pyautogui.center(self.editor_position)
            pyautogui.click(editor_center)
            
            # Select all existing code (Cmd+A on Mac)
            pyautogui.hotkey('command', 'a')
            time.sleep(0.5)
            
            # Type the new solution
            pyautogui.write(solution_code)
            return True
            
        except Exception as e:
            print(f"Error typing solution: {e}")
            return False

    async def analyze_code(self):
        """Background task that analyzes code when detected in the screen"""
        while True:
            if self.last_screen and not self.code_detected:
                self.code_detected = True
                
                if not self.analyzer_session:
                    self.analyzer_session = await client.aio.live.connect(
                        model=MODEL, 
                        config=CONFIG
                    ).__aenter__()
                    await self.analyzer_session.send(input=CODE_ANALYZER_PROMPT, end_of_turn=True)
                
                image_io = io.BytesIO()
                self.last_screen.save(image_io, format="jpeg")
                image_io.seek(0)
                image_io.seek(0)
                image_bytes = image_io.read()
                
                await self.analyzer_session.send(
                    input={"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()},
                    end_of_turn=True
                )
            
            await asyncio.sleep(2.0)

    def is_solution_request(self, text: str) -> bool:
        """Check if the text is requesting a solution."""
        solution_phrases = [
            "give me solution",
            "show me the code",
            "solve this",
            "how to solve",
            "what's the solution",
            "what is the solution",
            "code solution",
            "implement this",
            "write the code",
            "help me code",
            "solution code",
            "coding solution",
            "give me the solution",  
            "show the solution",
            "solve it",
            "solve the problem",
            "write solution",
            "solution please",
        ]
        text = text.lower().strip()
        return any(phrase in text or text in phrase for phrase in solution_phrases)

    def create_gui(self):
        """Create the GUI in a separate process"""
        try:
            # Start GUI in a new process
            gui_process = multiprocessing.Process(target=run_leetcode_gui)
            gui_process.daemon = True  # Make process exit when main process exits
            gui_process.start()
            self.gui_showing = True
        except Exception as e:
            print(f"Error launching GUI: {e}")
            self.gui_showing = False

    async def process_voice_command(self, command):
        """Process voice commands including requests to type solutions"""
        print(f"Processing command: {command}")  # Debug print
        
        if self.is_solution_request(command):
            print("Solution request detected, launching GUI...")  # Debug print
            # Launch solution GUI without sending to model
            if not self.gui_showing:
                # Create GUI in separate process
                self.create_gui()
            return  # Don't process further to avoid audio response
            
        elif "type solution" in command.lower() or "implement solution" in command.lower():
            # Keep existing solution typing functionality
            if not self.analyzer_session:
                print("No code detected yet, please wait...")
                return
                
            await self.analyzer_session.send(
                input="Please provide the complete solution code for this problem.",
                end_of_turn=True
            )
            
            # Get the response with the solution
            turn = self.analyzer_session.receive()
            async for response in turn:
                if text := response.text:
                    # Try to type the solution
                    if await self.type_solution(text):
                        await self.session.send(
                            input="I've typed the solution into the editor. Please review it before submitting.",
                            end_of_turn=True
                        )
                    else:
                        await self.session.send(
                            input="I couldn't type the solution. Please make sure the editor is visible.",
                            end_of_turn=True
                        )

    async def receive_audio(self):
        """Background task to read from the websocket and process audio/text"""
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")
                    # Process any voice commands
                    await self.process_voice_command(text)

            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def send_text(self):
        """Process text input from console"""
        while True:
            try:
                text = await asyncio.to_thread(
                    input,
                    "message > "
                )
                if text.lower() == "q":
                    break
                    
                # Check for solution request in text input
                if self.is_solution_request(text):
                    print("Text solution request detected")
                    await self.process_voice_command(text)
                else:
                    await self.session.send(input=text or ".", end_of_turn=True)
            except Exception as e:
                print(f"Error in send_text: {e}")
                await asyncio.sleep(1)  # Avoid tight loop on error

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.audio_out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        # Store the image for code analysis
        self.last_screen = img

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.audio_out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.audio_out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.audio_out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                # Send tutor prompt
                await self.session.send(input=TUTOR_PROMPT, end_of_turn=True)

                self.audio_in_queue = asyncio.Queue()
                self.audio_out_queue = asyncio.Queue(maxsize=5)

                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.get_screen())  # Always capture screen for LeetCode
                tg.create_task(self.analyze_code())  # Add code analysis task
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                tg.create_task(self.send_text())  # Add text input task

                # Keep running until interrupted
                while True:
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            if self.audio_stream:
                self.audio_stream.close()
            if self.analyzer_session:
                await self.analyzer_session.__aexit__(None, None, None)
            traceback.print_exception(EG)


if __name__ == "__main__":
    # Required for multiprocessing on macOS
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="Input mode (only screen mode supported)",
        choices=["screen"],
    )
    args = parser.parse_args()
    assistant = LeetCodeAssistant(video_mode=args.mode)
    asyncio.run(assistant.run())
