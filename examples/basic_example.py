#!/usr/bin/env python3
"""
Basic example of using Chain of Debate.

This example demonstrates the most basic usage of the Chain of Debate library
with a simple question and default settings.
"""

import os
from chain_of_debate import ChainOfDebate


def main():
    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set the OPENAI_API_KEY environment variable")
        return
    
    # Define a question
    question = """
    What are the key factors that contribute to the success of a startup company?
    Consider both internal factors (team, product, strategy) and external factors
    (market conditions, competition, funding).
    """
    
    print("Chain of Debate - Basic Example")
    print("=" * 50)
    print(f"Question: {question.strip()}")
    print("=" * 50)
    
    # Initialize Chain of Debate with basic settings
    cod = ChainOfDebate(
        n_debate_agents=3,
        max_rounds_per_debate=3,
        max_questions=5,
        verbose=True,
        progressbar=True
    )
    
    # Run the debate
    try:
        result = cod.run(question)
        
        print("\n" + "=" * 80)
        print("FINAL ANSWER:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        # Print debug information
        cod.print_debug_summary()
        
    except Exception as e:
        print(f"Error running Chain of Debate: {e}")


if __name__ == "__main__":
    main()
