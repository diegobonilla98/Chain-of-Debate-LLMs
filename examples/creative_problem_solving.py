#!/usr/bin/env python3
"""
Creative problem solving with high-creativity agents.

This example uses the "creative" agent configuration to generate innovative
solutions to open-ended problems.
"""

import os
from chain_of_debate import ChainOfDebate


def main():
    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set the OPENAI_API_KEY environment variable")
        return
    
    # Define a creative design challenge
    question = """
    Design an innovative urban transportation system for a city of 2 million people 
    that achieves these goals:
    
    - Reduces transportation emissions by 80% within 10 years
    - Provides equitable access to all socioeconomic groups
    - Integrates seamlessly with existing infrastructure
    - Remains economically viable and self-sustaining
    - Enhances quality of life and community connection
    
    Think beyond conventional solutions. Consider emerging technologies, 
    behavioral psychology, urban planning innovations, and sustainable financing models.
    """
    
    print("Chain of Debate - Creative Problem Solving Example")
    print("=" * 55)
    print(f"Question: {question.strip()}")
    print("=" * 55)
    
    # Initialize Chain of Debate with creative configuration
    cod = ChainOfDebate(
        n_debate_agents=3,
        max_rounds_per_debate=5,
        max_questions=7,
        agent_config_type="creative",  # Use high-creativity agents
        include_code_execution=True,   # For calculations and modeling
        include_web_search=True,       # For research on emerging tech
        verbose=True,
        progressbar=True,
        save_log=True
    )
    
    print("🚀 Using creative agent configuration for innovative solutions")
    print("🧠 High-temperature agents for unconventional thinking")
    print("🔍 Web search enabled for emerging technology research")
    print("📊 Code execution enabled for feasibility modeling")
    print("=" * 55)
    
    # Run the debate
    try:
        result = cod.run(question)
        
        print("\n" + "=" * 80)
        print("INNOVATIVE TRANSPORTATION SYSTEM DESIGN:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        # Print debug information
        cod.print_debug_summary()
        
        # Show log file location
        if cod.last_log_file:
            print(f"\nDetailed design process log saved to: {cod.last_log_file}")
        
    except Exception as e:
        print(f"Error running Chain of Debate: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
