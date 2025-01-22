from enum import Enum, auto
import asyncio
import logging
import pyaudio
from typing import Dict, Any, Optional
from google import genai

logger = logging.getLogger(__name__)

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
MODEL = "models/gemini-2.0-flash-exp"

class InterviewState(Enum):
    ASK_NAME = auto()
    INTRODUCTION = auto()
    ASK_PERMISSION = auto()
    PROBLEM_DESCRIPTION = auto()
    QUESTIONS = auto()
    CODING = auto()

PROMPT_MAPPING = {
    InterviewState.ASK_NAME: "Hello! I'm your technical interviewer today. Could you please tell me your name?",
    InterviewState.INTRODUCTION: (
        "Thank you, {candidate_name}. This will be a 25-minute technical interview where you'll solve a coding problem. "
        "Are you ready to begin?"
    ),
    InterviewState.PROBLEM_DESCRIPTION: (
        "Excellent. Today's problem is '{problem_name}'.\n\n"
        "{description}\n\n"
        "Constraints:\n{constraints}\n\n"
        "Examples:\n{examples}\n\n"
        "Do you have any questions about the problem?"
    ),
}

class AudioInterviewer:
    def __init__(self, problem_data: Dict[Any, Any], model: str = MODEL):
        self.problem_data = problem_data
        self.state = InterviewState.ASK_NAME
        self.candidate_name = None
        
        # Audio queues
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=5)
        
        # Audio streams
        self.audio_stream = None
        self.pya = pyaudio.PyAudio()
        
        # Gemini config
        self.config = {
            "generation_config": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "response_modalities": ["AUDIO"]
            },
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }
        
        # Initialize Gemini client with model
        self.client = genai.Client(http_options={"api_version": "v1alpha"})
        self.model = model
        self.session = None
        
    def _format_constraints(self) -> str:
        """Formats problem constraints into a bulleted list."""
        return "\n".join([f"- {c}" for c in self.problem_data['constraints']])
    
    def _format_examples(self) -> str:
        """Formats problem examples."""
        examples = []
        for i, example in enumerate(self.problem_data['test_cases'], start=1):
            examples.append(
                f"Example {i}:\n"
                f"Input: nums = {example['input']}, target = {example['target']}\n"
                f"Output: {example['expected_output']}\n"
                f"Explanation: {example['description']}\n"
            )
        return "\n".join(examples)

    def _get_interviewer_prompt(self) -> str:
        """Retrieve the appropriate prompt for the current state."""
        # Retrieve the template for the current state
        prompt_template = PROMPT_MAPPING.get(self.state, "")
        if not prompt_template:
            logger.warning(f"No prompt found for state {self.state}")
            return ""

        # Prepare context for dynamic formatting
        context = {
            "candidate_name": self.candidate_name,
            "problem_name": self.problem_data.get("name", "Unknown Problem"),
            "description": self.problem_data.get("description", "No description provided."),
            "constraints": self._format_constraints(),
            "examples": self._format_examples(),
        }

        try:
            return prompt_template.format(**context)
        except KeyError as e:
            logger.error(f"Missing context for prompt formatting: {e}")
            return "There was an error generating the prompt. Please wait."

    async def listen_audio(self):
        """Background task to listen for audio input"""
        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        
        while True:
            try:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE)
                await self.session.send(input={"data": data, "mime_type": "audio/pcm"})
            except Exception as e:
                logger.error(f"Error reading audio: {e}")
                await asyncio.sleep(0.1)

    async def play_audio(self):
        """Background task to play audio output"""
        stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        
        while True:
            try:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)
            except Exception as e:
                logger.error(f"Error playing audio: {e}")
                await asyncio.sleep(0.1)

    async def process_audio_response(self, text: str):
        """Process transcribed audio and update interview state"""
        logger.info(f"Processing response: {text}")
        text_lower = text.lower()
        
        if self.state == InterviewState.ASK_NAME:
            # Extract name from response
            self.candidate_name = text.strip()
            self.state = InterviewState.INTRODUCTION
            response = self._get_interviewer_prompt()
            logger.info(f"Sending response: {response}")
            await self.session.send(input=response, end_of_turn=True)
            
        elif self.state == InterviewState.INTRODUCTION:
            if any(word in text_lower for word in ['yes', 'ready', 'sure']):
                self.state = InterviewState.PROBLEM_DESCRIPTION
                response = self._get_interviewer_prompt()
                logger.info(f"Sending response: {response}")
                await self.session.send(input=response, end_of_turn=True)
                
        elif self.state == InterviewState.PROBLEM_DESCRIPTION:
            self.state = InterviewState.QUESTIONS
            response = "Do you have any questions about the problem description or constraints?"
            logger.info(f"Sending response: {response}")
            await self.session.send(input=response, end_of_turn=True)
            
        elif self.state == InterviewState.QUESTIONS:
            if "ready" in text_lower or "no" in text_lower:
                self.state = InterviewState.CODING
                response = "Great! You can start implementing your solution now. Feel free to think out loud and explain your approach as you code."
                logger.info(f"Sending response: {response}")
                await self.session.send(input=response, end_of_turn=True)
            else:
                response = await self._handle_question(text)
                logger.info(f"Sending response: {response}")
                await self.session.send(input=response, end_of_turn=True)

    async def receive_audio(self):
        """Background task to receive and process audio from Gemini"""
        while True:
            try:
                turn = self.session.receive()
                async for response in turn:
                    if data := response.data:
                        await self.audio_in_queue.put(data)
                    if text := response.text:
                        logger.info(f"Received text: {text}")
                        await self.process_audio_response(text)
            except Exception as e:
                logger.error(f"Error in receive_audio: {e}")
                await asyncio.sleep(0.1)

    async def _handle_question(self, question: str) -> str:
        """Handle candidate questions about the problem description."""
        question_lower = question.lower()
        
        # Handle questions about approach/complexity
        if any(word in question_lower for word in ['optimal', 'optimized', 'complexity', 'better solution', 'naive', 'brute force']):
            return ("You're free to implement any solution you think is best. If you know an optimized solution, feel free to implement that directly. "
                   "However, if you think implementing a simpler solution first will help you get to the optimized solution, that's perfectly fine too. "
                   "What matters is that you can explain your thought process.")
        
        # Handle questions about constraints
        if any(word in question_lower for word in ['negative', 'positive', 'zero', 'empty', 'null', 'range']):
            constraints = self.problem_data['constraints']
            relevant_constraints = [c for c in constraints if any(word in c.lower() for word in question_lower.split())]
            if relevant_constraints:
                return f"According to the constraints: {' '.join(relevant_constraints)}"
            
        # Handle questions about input format
        if any(word in question_lower for word in ['input', 'array', 'format']):
            return ("The input will be an array of integers 'nums' and an integer 'target'. "
                   f"The array length will be between {self.problem_data['constraints'][0].split('<=')[1]}, "
                   f"and the numbers in the array will be in the range specified in the constraints.")
        
        # Handle questions about output format
        if any(word in question_lower for word in ['output', 'return', 'indices']):
            return ("You should return two indices from the input array where the corresponding numbers sum to the target value. "
                   "The order of the indices in the output doesn't matter.")
        
        # Handle questions about duplicate values
        if any(word in question_lower for word in ['duplicate', 'same number', 'same element']):
            return ("You may use the same number from different indices, but you cannot use the same index twice. "
                   "As shown in the examples, if the same number appears multiple times in the array, "
                   "you can use it as long as you're using different indices.")
        
        # Default response encouraging clarity
        return "Could you please clarify your question? I want to make sure I understand what aspect of the problem you're asking about."

    async def run(self):
        """Main execution loop"""
        try:
            async with (
                self.client.aio.live.connect(model=self.model, config=self.config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                
                # Prepare problem-specific system instruction
                problem_intro = f"""You are conducting a technical interview for LeetCode Problem #{self.problem_data['number']}: {self.problem_data['name']}.
The problem description, constraints, and test cases are:

Problem Description:
{self.problem_data['description']}

Constraints:
{chr(10).join(f"- {c}" for c in self.problem_data['constraints'])}

Example Test Cases:
1. Input: nums = {self.problem_data['test_cases'][0]['input']}, target = {self.problem_data['test_cases'][0]['target']}
   Output: {self.problem_data['test_cases'][0]['expected_output']}
   Explanation: {self.problem_data['test_cases'][0]['description']}

2. Input: nums = {self.problem_data['test_cases'][1]['input']}, target = {self.problem_data['test_cases'][1]['target']}
   Output: {self.problem_data['test_cases'][1]['expected_output']}
   Explanation: {self.problem_data['test_cases'][1]['description']}"""

                system_instruction = f"""{problem_intro}

You are a professional technical interviewer conducting this coding interview. Follow these guidelines:
1. Start by asking for the candidate's name politely and professionally.
2. After getting the name, explain this is a 25-minute technical interview and ask if they're ready.
3. Once they confirm, present THIS SPECIFIC coding problem clearly with the examples provided above.
4. For clarifying questions:
   - Only answer questions about THIS problem's description, constraints, and test cases
   - Do not provide any information about potential solutions or their complexities
   - If asked about implementation approach, encourage them to implement their preferred solution
   - Refer to the specific constraints and examples above to clarify edge cases
5. Be concise, professional, and supportive throughout the interview.
6. Stay in character as the interviewer at all times.
7. ONLY discuss the Two Sum problem as specified in the problem data."""

                logger.info("Sending system instruction with problem data")
                await self.session.send(input=system_instruction, end_of_turn=True)
                
                # Start audio tasks
                tg.create_task(self.listen_audio())
                tg.create_task(self.play_audio())
                tg.create_task(self.receive_audio())
                
                # Send initial prompt after audio tasks are started
                initial_prompt = self._get_interviewer_prompt()
                logger.info(f"Sending initial prompt: {initial_prompt}")
                await self.session.send(input=initial_prompt, end_of_turn=True)
                
                # Keep running until we transition to coding state
                while self.state != InterviewState.CODING:
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Error during interview: {str(e)}")
            raise
        finally:
            if self.audio_stream:
                self.audio_stream.close()
