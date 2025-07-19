#!/usr/bin/env python3
"""
Data analysis example with code execution agent.

This example demonstrates using the code execution agent to analyze data
and provide data-driven insights as part of the debate process.
"""

import os
from chain_of_debate import ChainOfDebate


def main():
    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set the OPENAI_API_KEY environment variable")
        return
    
    # Define a data analysis question
    question = """
    A retail company has the following monthly sales data for the past 2 years:
    
    Year 1: [45000, 48000, 52000, 58000, 62000, 68000, 72000, 75000, 71000, 65000, 58000, 85000]
    Year 2: [50000, 53000, 57000, 63000, 68000, 74000, 78000, 81000, 76000, 70000, 63000, 92000]
    
    Analyze these sales trends and provide recommendations for:
    1. Sales forecasting for the next 6 months
    2. Identifying seasonal patterns
    3. Marketing strategy optimization
    4. Inventory planning considerations
    
    Use statistical analysis and data visualization where appropriate.
    """
    
    print("Chain of Debate - Data Analysis Example")
    print("=" * 50)
    print(f"Question: {question.strip()}")
    print("=" * 50)
    
    # Initialize Chain of Debate with code execution capabilities
    cod = ChainOfDebate(
        n_debate_agents=3,
        max_rounds_per_debate=4,
        max_questions=6,
        agent_config_type="default",  # Use default agents
        include_code_execution=True,  # Enable Python code execution
        include_web_search=False,     # Not needed for this example
        verbose=True,
        progressbar=True,
        save_log=True
    )
    
    print("🐍 Code execution agent enabled for data analysis")
    print("📊 Agents can run Python code for statistical analysis and visualization")
    print("=" * 50)
    
    # Run the debate
    try:
        result = cod.run(question)
        
        print("\n" + "=" * 80)
        print("DATA ANALYSIS & RECOMMENDATIONS:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        # Print debug information
        cod.print_debug_summary()
        
        # Show log file location
        if cod.last_log_file:
            print(f"\nDetailed analysis log saved to: {cod.last_log_file}")
        
    except Exception as e:
        print(f"Error running Chain of Debate: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
