#!/usr/bin/env python3
"""
Simple Chain of Debate Runner

A simple script that runs the question from question.txt through Chain of Debate
with default settings. No interactive prompts - just runs and outputs the result.
"""

import os
import time

from chain_of_debate import ChainOfDebate

os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"
# Check API key
assert os.getenv("OPENAI_API_KEY")

# Load question
with open("question.txt", 'r', encoding='utf-8') as file:
    question = file.read().strip()

print("Chain of Debate - Simple Runner")
print("=" * 50)
print("Question:", question, end="\n\n")

# Initialize and run
try:
    print("🚀 Starting Chain of Debate (default agents, verbose logging)...")
    start_time = time.time()

    cod = ChainOfDebate(
        n_debate_agents=3,
        max_rounds_per_debate=5,
        max_questions=5,
        leader_model="gpt-4.1-nano",
        oracle_model="gpt-4.1-nano",
        debate_model="gpt-4.1-nano",
        debate_agent_models=None,
        leader_temperature=0.5,
        oracle_temperature=0.2,
        max_tokens_per_response=1024,
        max_tokens_final_response=1024,
        top_p=0.95,
        verbose=True,
        progressbar=True,
        save_log=True,
        debate_agent_configs=None,
        agent_config_type="default",
        include_code_execution=False,
        include_web_search=False
    )
    
    result = cod.run(question)
    
    end_time = time.time()
    
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(result)
    print("=" * 80)
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    # Save result
    with open("answer.txt", 'w', encoding='utf-8') as file:
        file.write(result)
    print("Result saved to answer.txt")
    
except Exception as e:
    print(f"Error: {e}")
