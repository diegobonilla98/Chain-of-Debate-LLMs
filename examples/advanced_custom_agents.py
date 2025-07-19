#!/usr/bin/env python3
"""
Advanced example with custom agent configurations.

This example shows how to create custom debate agents with specific personalities
and expertise areas, and how to use special agents (code execution and web search).
"""

import os
from chain_of_debate import ChainOfDebate


def main():
    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set the OPENAI_API_KEY environment variable")
        return
    
    # Define a complex technical question
    question = """
    Design a comprehensive cybersecurity strategy for a mid-size financial services 
    company (500 employees) that handles sensitive customer data. The strategy should 
    address technical controls, employee training, incident response, and compliance 
    with financial regulations. Consider both current threats and emerging risks.
    """
    
    # Custom agent configurations for cybersecurity expertise
    custom_agents = [
        {
            "name": "Security Architect",
            "description": "Expert in cybersecurity frameworks, technical controls, and enterprise security architecture. Focuses on preventive measures and defense-in-depth strategies.",
            "temperature": 0.4
        },
        {
            "name": "Compliance Specialist",
            "description": "Expert in financial regulations, privacy laws, and compliance frameworks like SOX, GDPR, and PCI-DSS. Emphasizes regulatory requirements and audit considerations.",
            "temperature": 0.3
        },
        {
            "name": "Incident Response Expert",
            "description": "Specialist in threat hunting, incident response, and crisis management. Focuses on detection, response procedures, and business continuity.",
            "temperature": 0.6
        },
        {
            "name": "Risk Assessment Analyst",
            "description": "Expert in risk analysis, threat modeling, and security metrics. Balances security investments with business risk tolerance.",
            "temperature": 0.5
        }
    ]
    
    print("Chain of Debate - Advanced Cybersecurity Example")
    print("=" * 60)
    print(f"Question: {question.strip()}")
    print("=" * 60)
    print(f"Using {len(custom_agents)} custom security experts")
    print("Agents:", ", ".join([agent["name"] for agent in custom_agents]))
    print("=" * 60)
    
    # Initialize Chain of Debate with advanced settings
    cod = ChainOfDebate(
        n_debate_agents=len(custom_agents),
        max_rounds_per_debate=4,
        max_questions=8,
        leader_model="gpt-4",
        oracle_model="gpt-4", 
        debate_model="gpt-4",
        leader_temperature=0.3,
        oracle_temperature=0.2,
        debate_agent_configs=custom_agents,
        include_code_execution=False,  # Not needed for this example
        include_web_search=True,       # Enable web search for current threat intelligence
        verbose=True,
        progressbar=True,
        save_log=True
    )
    
    # Run the debate
    try:
        result = cod.run(question)
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE CYBERSECURITY STRATEGY:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        # Print debug information
        cod.print_debug_summary()
        
        # Show log file location
        if cod.last_log_file:
            print(f"\nDetailed debate log saved to: {cod.last_log_file}")
        
    except Exception as e:
        print(f"Error running Chain of Debate: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
