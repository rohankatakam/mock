from enum import Enum, auto
from typing import Optional, Dict, Any
import asyncio
import logging

logger = logging.getLogger(__name__)
MODEL = "models/gemini-2.0-flash-exp"

class InterviewState(Enum):
    ASK_NAME = auto()
    INTRODUCTION = auto()
    ASK_PERMISSION = auto()
    PROBLEM_DESCRIPTION = auto()
    QUESTIONS = auto()
    CODING = auto()

class AudioInterviewer(AudioLoop):
    def __init__(self, problem_data: Dict[Any, Any], model: str = MODEL):
        config = {
            "model": model,
            "generation_config": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "response_modalities": ["AUDIO"]
            },
            "system_instruction": """You are a professional technical interviewer conducting a coding interview. Follow these guidelines:
1. Start by asking for the candidate's name politely and professionally.
2. After getting the name, explain this is a 25-minute technical interview and ask if they're ready.
3. Once they confirm, present the coding problem clearly with examples.
4. For clarifying questions:
   - Only answer questions about the problem description, constraints, and test cases
   - Do not provide any information about potential solutions or their complexities
   - If asked about implementation approach, encourage them to implement their preferred solution
   - Refer to the problem constraints and examples to clarify edge cases
5. Be concise, professional, and supportive throughout the interview.
6. Stay in character as the interviewer at all times."""
        }
        super().__init__(config=config)
        self.problem_data = problem_data
        self.state = InterviewState.ASK_NAME
        self.candidate_name = None

    def _get_interviewer_prompt(self) -> str:
        """Generate the appropriate prompt based on current state."""
        if self.state == InterviewState.ASK_NAME:
            return "Hello! I'm your technical interviewer today. Could you please tell me your name?"
        elif self.state == InterviewState.INTRODUCTION:
            return f"Thank you, {self.candidate_name}. This will be a 25-minute technical interview where you'll solve a coding problem. Are you ready to begin?"
        elif self.state == InterviewState.PROBLEM_DESCRIPTION:
            constraints = "\n".join([f"- {c}" for c in self.problem_data['constraints']])
            return (
                f"Excellent. Today's problem is '{self.problem_data['name']}'.\n\n"
                f"{self.problem_data['description']}\n\n"
                f"Constraints:\n{constraints}\n\n"
                "Let me walk you through some examples:\n"
                f"Example 1:\n"
                f"Input: nums = {self.problem_data['test_cases'][0]['input']}, target = {self.problem_data['test_cases'][0]['target']}\n"
                f"Output: {self.problem_data['test_cases'][0]['expected_output']}\n"
                f"Explanation: {self.problem_data['test_cases'][0]['description']}\n\n"
                f"Example 2:\n"
                f"Input: nums = {self.problem_data['test_cases'][1]['input']}, target = {self.problem_data['test_cases'][1]['target']}\n"
                f"Output: {self.problem_data['test_cases'][1]['expected_output']}\n"
                f"Explanation: {self.problem_data['test_cases'][1]['description']}\n\n"
                "Do you have any questions about the problem?"
            )
        return ""

    async def _iter(self):
        """Override _iter to handle stateful interview flow."""
        print("Interview started. Type 'q' to quit at any time.")
        
        # Initial greeting - ask for name
        yield self._get_interviewer_prompt()
        
        while True:
            # Get candidate's response
            text = await asyncio.to_thread(input, "Candidate > ")
            
            if text.lower() == 'q':
                break
                
            # Update state based on response
            if self.state == InterviewState.ASK_NAME:
                self.candidate_name = text.strip()
                self.state = InterviewState.INTRODUCTION
                yield self._get_interviewer_prompt()
            elif self.state == InterviewState.INTRODUCTION:
                if any(word in text.lower() for word in ['yes', 'ready', 'sure']):
                    self.state = InterviewState.PROBLEM_DESCRIPTION
                    yield self._get_interviewer_prompt()
            elif self.state == InterviewState.PROBLEM_DESCRIPTION:
                self.state = InterviewState.QUESTIONS
                yield "Do you have any questions about the problem description or constraints?"
            elif self.state == InterviewState.QUESTIONS:
                if "ready" in text.lower() or "no" in text.lower():
                    yield "Great! You can start implementing your solution now. Feel free to think out loud and explain your approach as you code."
                    self.state = InterviewState.CODING
                else:
                    response = await self._handle_question(text)
                    yield response

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

# Example usage
async def run_audio_interview():
    interviewer = AudioInterviewer(problem_data=lc_data)
    await interviewer.run()