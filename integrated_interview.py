from enum import Enum, auto
import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional
from audio_interview import AudioInterviewer, InterviewState
from leetcode_assistant import LeetCodeAssistant, MODEL, CONFIG

logger = logging.getLogger(__name__)

class IntegratedInterviewState(Enum):
    INITIAL_INTERVIEW = auto()  # Using AudioInterviewer
    LEETCODE_ASSISTANT = auto() # Using LeetCodeAssistant

class IntegratedInterviewer:
    def __init__(self):
        # Load problem data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'lc_data.json')
        with open(data_path, 'r') as f:
            self.problem_data = json.load(f)
            
        # Initialize components
        self.interview_state = IntegratedInterviewState.INITIAL_INTERVIEW
        self.audio_interviewer = AudioInterviewer(problem_data=self.problem_data)
        self.leetcode_assistant = None  # Will be initialized when needed
        
    async def transition_to_leetcode(self):
        """Transition from interview phase to LeetCode coding phase"""
        logger.info("Transitioning to LeetCode phase")
        self.interview_state = IntegratedInterviewState.LEETCODE_ASSISTANT
        self.leetcode_assistant = LeetCodeAssistant(video_mode="screen")
        
        # Start LeetCode assistant with its own audio handling
        await self.leetcode_assistant.run()
        
    async def run(self):
        """Main execution loop"""
        try:
            # Start with interview phase
            await self.audio_interviewer.run()
            
            # Once interview phase is complete (state == CODING), transition to LeetCode
            if self.audio_interviewer.state == InterviewState.CODING:
                await self.transition_to_leetcode()
                
        except Exception as e:
            logger.error(f"Error during interview: {str(e)}")
            raise

async def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    interviewer = IntegratedInterviewer()
    await interviewer.run()

if __name__ == "__main__":
    asyncio.run(main())
