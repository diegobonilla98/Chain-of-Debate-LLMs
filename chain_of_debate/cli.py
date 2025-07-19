#!/usr/bin/env python3
"""
Command Line Interface for Chain of Debate.

Provides a command-line interface to run Chain of Debate sessions with various
configuration options and input methods.
"""

import argparse
import os
import sys
import time
from typing import Optional

from .core import ChainOfDebate


def setup_api_key():
    """Ensure OpenAI API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    return api_key


def load_question_from_file(file_path: str) -> str:
    """Load question from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            question = f.read().strip()
        if not question:
            raise ValueError("Question file is empty")
        return question
    except FileNotFoundError:
        print(f"Error: Question file '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading question file: {e}")
        sys.exit(1)


def save_result(result: str, output_file: Optional[str] = None):
    """Save the result to a file."""
    if output_file is None:
        output_file = "answer.txt"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Result saved to: {output_file}")
    except Exception as e:
        print(f"Warning: Could not save result to file: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Chain of Debate: Collaborative AI problem solving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with question from file
  chain-of-debate --question-file question.txt
  
  # Run with direct question
  chain-of-debate --question "What is the best approach to climate change?"
  
  # Use creative agents with special capabilities
  chain-of-debate --question "Design a new product" --agent-config creative --code-execution --web-search
  
  # Quiet mode with custom output
  chain-of-debate --question-file question.txt --quiet --output result.txt
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--question", "-q",
        type=str,
        help="Question to ask the debate system"
    )
    input_group.add_argument(
        "--question-file", "-f",
        type=str,
        help="Path to text file containing the question"
    )
    
    # Agent configuration
    parser.add_argument(
        "--agent-config", "-c",
        type=str,
        choices=["default", "conservative", "creative"],
        default="default",
        help="Type of agent configuration to use (default: default)"
    )
    
    parser.add_argument(
        "--n-agents", "-n",
        type=int,
        default=3,
        help="Number of debate agents (default: 3)"
    )
    
    # Model configuration
    parser.add_argument(
        "--leader-model",
        type=str,
        default="gpt-4.1",
        help="Model for the leader agent (default: gpt-4.1)"
    )
    
    parser.add_argument(
        "--oracle-model",
        type=str,
        default="gpt-4.1",
        help="Model for the oracle agent (default: gpt-4.1)"
    )
    
    parser.add_argument(
        "--debate-model",
        type=str,
        default="gpt-4.1",
        help="Model for debate agents (default: gpt-4.1)"
    )
    
    # Debate parameters
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum debate rounds per question (default: 5)"
    )
    
    parser.add_argument(
        "--max-questions",
        type=int,
        default=5,
        help="Maximum questions the leader can ask (default: 5)"
    )
    
    # Special agents
    parser.add_argument(
        "--code-execution",
        action="store_true",
        help="Include code execution agent"
    )
    
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Include web search agent"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for the result (default: answer.txt)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars and non-essential output"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save detailed process log"
    )
    
    # Temperature settings
    parser.add_argument(
        "--leader-temperature",
        type=float,
        default=0.5,
        help="Temperature for leader agent (default: 0.5)"
    )
    
    parser.add_argument(
        "--oracle-temperature",
        type=float,
        default=0.2,
        help="Temperature for oracle agent (default: 0.2)"
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup API key
    setup_api_key()
    
    # Load question
    if args.question:
        question = args.question
    else:
        question = load_question_from_file(args.question_file)
    
    # Validate question
    if not question or len(question.strip()) < 10:
        print("Error: Question is too short. Please provide a meaningful question.")
        sys.exit(1)
    
    # Print header unless quiet
    if not args.quiet:
        print("Chain of Debate - CLI")
        print("=" * 50)
        print(f"Question: {question}")
        print("=" * 50)
        print()
    
    try:
        # Initialize Chain of Debate
        if not args.quiet:
            print("🚀 Initializing Chain of Debate...")
        
        cod = ChainOfDebate(
            n_debate_agents=args.n_agents,
            max_rounds_per_debate=args.max_rounds,
            max_questions=args.max_questions,
            leader_model=args.leader_model,
            oracle_model=args.oracle_model,
            debate_model=args.debate_model,
            leader_temperature=args.leader_temperature,
            oracle_temperature=args.oracle_temperature,
            verbose=args.verbose and not args.quiet,
            progressbar=not args.quiet,
            save_log=not args.no_save,
            agent_config_type=args.agent_config,
            include_code_execution=args.code_execution,
            include_web_search=args.web_search
        )
        
        # Run the debate
        start_time = time.time()
        result = cod.run(question)
        end_time = time.time()
        
        # Print results
        if not args.quiet:
            print("\n" + "=" * 80)
            print("FINAL ANSWER:")
            print("=" * 80)
        
        print(result)
        
        if not args.quiet:
            print("=" * 80)
            print(f"Execution time: {end_time - start_time:.2f} seconds")
            
            # Print debug summary
            cod.print_debug_summary()
        
        # Save result
        save_result(result, args.output)
        
        # Print log file location
        if not args.no_save and cod.last_log_file and not args.quiet:
            print(f"Detailed log saved to: {cod.last_log_file}")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
