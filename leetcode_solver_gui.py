#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI version of LeetCode solver using Gemini API.
Features:
- Text area for problem input
- Button to generate solution
- Text area for solution output with syntax highlighting
- Copy button for easy solution copying
"""

import asyncio
import re
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import ttkthemes
from google import genai

# Use the Gemini model for generating code
MODEL = "models/gemini-2.0-flash-exp"
CONFIG = {"generation_config": {"response_modalities": ["TEXT"]}}

PROMPT = """You are a LeetCode solution expert. Given this problem, provide ONLY the complete Python solution code that will pass all test cases. 
Format your response exactly like this, with no other text or explanations:
```python
class Solution:
    def methodName(self, param1: type1, param2: type2) -> returnType:
        # Solution implementation
        return result
```"""

def extract_code(text: str) -> str:
    """Extract code from between ```python and ``` markers."""
    match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    return match.group(1).strip() if match else text

class LeetCodeSolverGUI:
    def __init__(self, root):
        self.root = root
        root.title("LeetCode Solution Generator")
        
        # Apply modern theme
        style = ttkthemes.ThemedStyle(root)
        style.set_theme("arc")  # Modern, clean theme
        
        # Configure grid weight
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(3, weight=1)
        
        # Problem input section
        ttk.Label(root, text="Enter LeetCode Problem Description:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.problem_text = scrolledtext.ScrolledText(root, height=10, width=80, wrap=tk.WORD)
        self.problem_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Generate button
        self.generate_btn = ttk.Button(root, text="Generate Solution", command=self.generate_solution)
        self.generate_btn.grid(row=2, column=0, pady=10)
        
        # Solution output section
        ttk.Label(root, text="Solution (ready to copy):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.solution_text = scrolledtext.ScrolledText(root, height=10, width=80)
        self.solution_text.grid(row=4, column=0, sticky="nsew", padx=5, pady=5)
        
        # Copy button
        self.copy_btn = ttk.Button(root, text="Copy Solution", command=self.copy_solution, state="disabled")
        self.copy_btn.grid(row=5, column=0, pady=10)
        
        # Status label
        self.status_label = ttk.Label(root, text="")
        self.status_label.grid(row=6, column=0, pady=5)
        
    async def get_solution(self, problem: str) -> str:
        """Get solution from Gemini API."""
        try:
            client = genai.Client(http_options={"api_version": "v1alpha"})
            
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                # Send prompt and problem together to reduce round trips
                await session.send(
                    input=f"{PROMPT}\n\nProblem:\n{problem}",
                    end_of_turn=True
                )
                
                # Get response and extract code with timeout
                solution = ""
                try:
                    async def get_response():
                        nonlocal solution
                        async for response in session.receive():
                            if response.text:
                                solution += response.text
                    
                    await asyncio.wait_for(get_response(), timeout=10)
                    return extract_code(solution)
                    
                except asyncio.TimeoutError:
                    return "Error: Request timed out after 10 seconds. Please try again."
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_solution(self):
        """Handle generate button click."""
        problem = self.problem_text.get("1.0", tk.END).strip()
        if not problem:
            self.status_label.config(text="Please enter a problem description")
            return
        
        # Clear previous solution
        self.solution_text.delete("1.0", tk.END)
        self.copy_btn.config(state="disabled")
        self.status_label.config(text="Generating solution...")
        self.generate_btn.config(state="disabled")
        self.root.update()  # Force UI update
        
        # Create a new event loop for this request
        async def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                solution = await self.get_solution(problem)
                # Update GUI in main thread
                self.root.after(0, self.update_solution, solution)
            finally:
                loop.close()
        
        # Run in a separate thread to not block GUI
        import threading
        threading.Thread(target=lambda: asyncio.run(run_async())).start()
    
    def update_solution(self, solution: str):
        """Update solution text and UI state."""
        self.solution_text.delete("1.0", tk.END)
        self.solution_text.insert("1.0", solution)
        self.generate_btn.config(state="normal")
        self.copy_btn.config(state="normal")  # Enable copy button
        self.status_label.config(text="Solution generated!")
    
    def copy_solution(self):
        """Copy solution to clipboard."""
        solution = self.solution_text.get("1.0", tk.END).strip()
        if solution:
            self.root.clipboard_clear()
            self.root.clipboard_append(solution)
            self.status_label.config(text="Solution copied to clipboard!")
            
            # Flash the solution text widget to give visual feedback
            orig_bg = self.solution_text.cget("background")
            self.solution_text.configure(background="lightgreen")
            self.root.after(200, lambda: self.solution_text.configure(background=orig_bg))
    
def main():
    root = tk.Tk()
    app = LeetCodeSolverGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
