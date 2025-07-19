import openai
import os
from typing import List, Dict, Optional
import difflib
from collections import Counter
import yaml
import json
import time
from datetime import datetime
from openai_cost_calculator import estimate_cost
from tqdm import tqdm
import asyncio
from openai import AsyncOpenAI
import io
import sys
import builtins
from contextlib import redirect_stdout, redirect_stderr

# ANSI color codes for logging
class Colors:
    RESET = "\033[0m"
    BLUE = "\033[94m"    # Questions
    GREEN = "\033[92m"   # Agreements, agent turns (proposals)
    RED = "\033[91m"     # Disagreements
    YELLOW = "\033[93m"  # Reflections, feedback
    PURPLE = "\033[95m"  # Modified answers, consensus
    CYAN = "\033[96m"    # Leader questions
    WHITE = "\033[97m"   # General debug

# Code execution functionality for agents
def execute_python_code(code: str) -> str:
    """
    Execute Python code safely and return the output or error.
    
    Security Features:
    - Whitelisted imports: Only allows safe standard library modules
    - Blocked dangerous functions: exec, eval, compile, open, input, breakpoint
    - Safe execution environment with restricted globals
    
    Allowed modules include: math, itertools, collections, datetime, json, 
    pandas, numpy, matplotlib, and other common safe modules.
    
    Args:
        code: Python code string to execute
        
    Returns:
        String containing the output, error, or "No output."
    """
    # Safe standard library modules that are allowed
    safe_modules = {
        'math', 'random', 'datetime', 'time', 'itertools', 'collections', 
        'functools', 'operator', 'copy', 'json', 'csv', 're', 'string',
        'statistics', 'fractions', 'decimal', 'typing', 'enum', 'dataclasses',
        'heapq', 'bisect', 'array', 'struct', 'weakref', 'types', 'abc',
        'contextlib', 'warnings', 'textwrap', 'unicodedata', 'locale',
        'calendar', 'hashlib', 'hmac', 'secrets', 'uuid', 'base64',
        'binascii', 'codecs', 'io', 'sys', 'gc', 'inspect', 'dis',
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'plotly', 'scipy'
    }
    
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Safe import function that only allows whitelisted modules."""
        if name.split('.')[0] in safe_modules:
            return __builtins__['__import__'](name, globals, locals, fromlist, level)
        else:
            raise ImportError(f"Import of '{name}' is not allowed for security reasons")
    
    # Safe builtins: Exclude dangerous functions but keep __import__ with our safe version
    dangerous = {'open', 'exec', 'eval', 'compile', 'input', 'breakpoint'}
    safe_builtins = {k: v for k, v in builtins.__dict__.items() 
                    if k not in dangerous and not k.startswith('__')}
    
    # Add our safe import function
    safe_builtins['__import__'] = safe_import
    
    output = io.StringIO()
    error = io.StringIO()
    
    try:
        safe_globals = {"__builtins__": safe_builtins}
        with redirect_stdout(output), redirect_stderr(error):
            exec(code, safe_globals, {})
        
        result = output.getvalue().strip()
        if not result:
            error_result = error.getvalue().strip()
            return error_result if error_result else "No output."
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"

# Function calling tool definitions
def get_code_execution_tools():
    """Get the tools definition for code execution."""
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": "Execute Python code and return the printed output or error. Use print() for results. Safe execution environment with limited builtins.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string", 
                            "description": "Python code to execute. Use print() to display results. Limited to safe operations."
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    ]

def get_web_search_tools():
    """Get the tools definition for web search (placeholder - would need real implementation)."""
    return [
        {
            "type": "function", 
            "function": {
                "name": "web_search",
                "description": "Search the web for current information and return relevant results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant information"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

def web_search_placeholder(query: str, num_results: int = 5) -> str:
    """
    Placeholder for web search functionality.
    In a real implementation, this would use a search API.
    """
    return f"[Web Search Placeholder] Query: '{query}' - This would return {num_results} search results in a real implementation. Consider integrating with search APIs like Google Custom Search, Bing, or SerpAPI."

class ChainOfDebate:
    def __init__(
        self,
        n_debate_agents: int = 3,
        max_rounds_per_debate: int = 5,
        max_questions: int = 10,
        leader_model: str = "gpt-4.1",
        oracle_model: str = "gpt-4.1",
        debate_model: str = "gpt-4.1",
        debate_agent_models: Optional[Dict[str, str]] = None,
        leader_temperature: float = 0.5,
        oracle_temperature: float = 0.2,
        max_tokens_per_response: int = 512,
        max_tokens_final_response: int = 1024,
        top_p: float = 0.95,
        verbose: bool = False,
        progressbar: bool = True,
        save_log: bool = True,
        use_async: bool = True,
        debate_agent_configs: Optional[List[Dict]] = None,
        agent_config_type: str = "default",
        include_code_execution: bool = False,
        include_web_search: bool = False
    ):
        """
        Initializes the Chain of Debate system.

        Args:
            n_debate_agents: Number of debate agents.
            max_rounds_per_debate: Max debate rounds per leader question.
            max_questions: Max questions the leader can ask.
            leader_model: Model for the leader agent.
            oracle_model: Model for the oracle agent.
            debate_model: Default model for debate agents.
            debate_agent_models: Dict to override model per debate agent name.
            leader_temperature: Temperature for the leader.
            oracle_temperature: Temperature for the oracle.
            max_tokens_per_response: Token limit per response.
            max_tokens_final_response: Token limit for final oracle response.
            top_p: Top-p sampling.
            verbose: Enable/disable logging.
            progressbar: Enable/disable progress bar display.
            save_log: Enable/disable saving the complete process log to a text file.
            use_async: Enable/disable asynchronous API calls for improved performance.
            debate_agent_configs: Configs for debate agents (overrides agent_config_type).
            agent_config_type: Type of agent configuration to load ("default", "conservative", "creative").
            include_code_execution: Whether to include a code execution agent.
            include_web_search: Whether to include a web search agent.
        """
        self.n_debate_agents = n_debate_agents
        self.max_rounds_per_debate = max_rounds_per_debate
        self.max_questions = max_questions
        self.leader_model = leader_model
        self.oracle_model = oracle_model
        self.debate_model = debate_model
        self.debate_agent_models = debate_agent_models or {}
        self.leader_temperature = leader_temperature
        self.oracle_temperature = oracle_temperature
        self.max_tokens_per_response = max_tokens_per_response
        self.max_tokens_final_response = max_tokens_final_response
        self.top_p = top_p
        self.verbose = verbose
        self.progressbar = progressbar
        self.save_log = save_log
        self.use_async = use_async
        self.agent_config_type = agent_config_type
        self.include_code_execution = include_code_execution
        self.include_web_search = include_web_search

        # Load agent configurations from JSON
        self._load_agent_configs()
        
        # Set debate agent configs
        if debate_agent_configs is None:
            self.debate_agent_configs = self._get_agent_configs_for_type(self.agent_config_type, n_debate_agents)
        else:
            self.debate_agent_configs = debate_agent_configs[:n_debate_agents]
        
        # Add special agents if requested
        if self.include_code_execution:
            if 'special_agents' in self.agent_configs and 'code_execution' in self.agent_configs['special_agents']:
                code_agent = self.agent_configs['special_agents']['code_execution'].copy()
            else:
                # Fallback if config not found
                code_agent = {
                    "name": "Code Execution Analyst",
                    "description": "Python programmer who uses code execution to analyze problems and provide data-driven insights through computational analysis. Make sure the Python scripts are small and output little text because context windows limitations. Use python code analysis as a tool to calculate useful stuff, not to develop an app that needs to be executed and outputs anything that is not plain text.",
                    "temperature": 0.7
                }
            # Add the correct tools format
            code_agent["tools"] = get_code_execution_tools()
            self.debate_agent_configs.append(code_agent)
        
        if self.include_web_search:
            if 'special_agents' in self.agent_configs and 'web_search' in self.agent_configs['special_agents']:
                web_agent = self.agent_configs['special_agents']['web_search'].copy()
            else:
                # Fallback if config not found
                web_agent = {
                    "name": "Web Research Specialist",
                    "description": "Information researcher who uses web search to find current data, external perspectives, and real-world context.",
                    "temperature": 0.8
                }
            # Add the correct tools format
            web_agent["tools"] = get_web_search_tools()
            self.debate_agent_configs.append(web_agent)

        # Set your OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Load prompts from YAML file
        self._load_prompts()

        # Team context
        agent_descriptions = "\n".join([f"- {cfg['name']}: {cfg['description']}" for cfg in self.debate_agent_configs])
        self.team_context = self.prompts['team_context'].format(agent_descriptions=agent_descriptions)
        
        # Debugging and tracking variables
        self.debug_info = {
            'total_time': 0.0,
            'total_agreements': 0,
            'total_decisions': 0,
            'agreement_rate': 0.0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0,
            'api_calls': 0,
            'cost_breakdown': {}
        }
        
        # Process log for saving to file
        self.process_log = []
        self.last_log_file = None  # Store path to the last saved log file

    def _load_prompts(self):
        """Load prompts from the YAML configuration file."""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'prompts.yaml')
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                self.prompts = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompts configuration file not found at {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def _load_agent_configs(self):
        """Load agent configurations from the JSON configuration file."""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'agents.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                self.agent_configs = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Agent configuration file not found at {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file: {e}")

    def _get_agent_configs_for_type(self, config_type: str, n_agents: int) -> List[Dict]:
        """Get agent configurations for the specified type and number of agents."""
        if config_type == "default":
            base_configs = self.agent_configs["default_agent_configs"]
        elif config_type in self.agent_configs["alternative_configs"]:
            base_configs = self.agent_configs["alternative_configs"][config_type]
        else:
            raise ValueError(f"Unknown agent config type: {config_type}. Available types: default, {', '.join(self.agent_configs['alternative_configs'].keys())}")
        
        # Repeat the base configs to match the requested number of agents
        configs = []
        for i in range(n_agents):
            configs.append(base_configs[i % len(base_configs)].copy())
        
        return configs

    async def _generate_response_async(self, api_params: Dict) -> Dict:
        """Generate an async OpenAI API response with rate limit handling."""
        import re
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        while True:
            try:
                return await client.chat.completions.create(**api_params)
            except Exception as e:
                error_str = str(e)
                # Check for rate limit error (429)
                if "429" in error_str and "rate_limit_exceeded" in error_str:
                    # Extract wait time from error message using regex
                    wait_match = re.search(r'Please try again in (\d+\.?\d*)s', error_str)
                    if wait_match:
                        wait_time = float(wait_match.group(1)) + 5.0  # Add 5 seconds as requested
                        self.log(f"Rate limit reached. Waiting {wait_time:.1f} seconds before retrying...", Colors.YELLOW)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Fallback if we can't parse wait time
                        self.log("Rate limit reached. Waiting 35 seconds before retrying...", Colors.YELLOW)
                        await asyncio.sleep(35.0)
                        continue
                else:
                    # Re-raise non-rate-limit errors
                    raise e

    async def _generate_debate_responses_async(self, debate_configs: List[Dict], current_question: str, 
                                              question: str, debate_history: List[str], 
                                              is_reflection_round: bool) -> List[Dict]:
        """Generate all debate agent responses asynchronously."""
        tasks = []
        
        for idx, cfg in enumerate(debate_configs):
            agent_name = cfg['name']
            agent_model = self.debate_agent_models.get(agent_name, self.debate_model)

            # Customize prompt based on agent type
            if agent_name == "Code Execution Analyst" and 'code_execution_prompt_template' in self.prompts:
                prompt_template = self.prompts['code_execution_prompt_template']
            elif agent_name == "Web Research Specialist" and 'web_search_prompt_template' in self.prompts:
                prompt_template = self.prompts['web_search_prompt_template']
            else:
                prompt_template = self.prompts['debate_base_prompt_template']
            
            debate_system_prompt = prompt_template.format(
                team_context=self.team_context,
                agent_name=cfg['name'],
                agent_description=cfg['description'],
                current_question=current_question,
                question=question,
                history="\n".join(debate_history)
            )
            if is_reflection_round:
                debate_system_prompt += "\n" + self.prompts['reflection_round_instruction']

            full_debate_prompt = "\n".join(debate_history) + f"\n\n{self.prompts['user_messages']['debate_agent_turn'].format(agent_name=cfg['name'])}"

            # Generate response parameters
            api_params = {
                "model": agent_model,
                "messages": [
                    {"role": "system", "content": debate_system_prompt},
                    {"role": "user", "content": full_debate_prompt}
                ],
                "temperature": cfg['temperature'],
                "max_tokens": self.max_tokens_per_response,
                "top_p": self.top_p
            }
            
            # Add tools if this agent has them
            if 'tools' in cfg and cfg['tools']:
                api_params['tools'] = cfg['tools']
                api_params['tool_choice'] = "auto"
            
            # Add metadata for processing
            task_data = {
                'api_params': api_params,
                'agent_config': cfg,
                'agent_model': agent_model,
                'agent_index': idx
            }
            
            tasks.append(self._generate_single_debate_response_async(task_data))
        
        # Execute all tasks concurrently
        responses = await asyncio.gather(*tasks)
        return responses

    async def _generate_single_debate_response_async(self, task_data: Dict) -> Dict:
        """Generate a single debate response asynchronously with metadata."""
        api_params = task_data['api_params']
        cfg = task_data['agent_config']
        agent_model = task_data['agent_model']
        agent_index = task_data['agent_index']
        
        try:
            # Generate the response
            response_obj = await self._generate_response_async(api_params)
            
            # Track API call costs
            if hasattr(response_obj, 'usage') and response_obj.usage:
                self._track_api_call(
                    agent_model,
                    response_obj.usage.prompt_tokens,
                    response_obj.usage.completion_tokens,
                    response_obj
                )
            
            # Extract response content and handle tool calls if present
            choice = response_obj.choices[0]
            debate_response = choice.message.content or ""
            
            # Handle tool calls (for code execution and web search)
            if choice.message.tool_calls:
                tool_results = []
                for tool_call in choice.message.tool_calls:
                    function_name = tool_call.function.name
                    
                    try:
                        # Parse function arguments with error handling
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        self.log(f"Warning: Malformed JSON in tool call for {function_name}: {e}", Colors.YELLOW)
                        self.log(f"Raw arguments: {tool_call.function.arguments}", Colors.YELLOW)
                        tool_results.append(f"Error: Malformed JSON in {function_name} tool call - {str(e)}")
                        continue
                    
                    if function_name == "execute_python":
                        code = function_args.get("code", "")
                        if not code:
                            tool_results.append("Error: No code provided to execute")
                            continue
                        result = execute_python_code(code)
                        tool_results.append(f"Code executed:\n```python\n{code}\n```\nResult: {result}")
                        
                    elif function_name == "web_search":
                        query = function_args.get("query", "")
                        if not query:
                            tool_results.append("Error: No search query provided")
                            continue
                        num_results = function_args.get("num_results", 5)
                        result = web_search_placeholder(query, num_results)
                        tool_results.append(f"Web search for '{query}': {result}")
                        
                    else:
                        tool_results.append(f"Unknown tool: {function_name}")
                
                # Append tool results to the response
                if tool_results:
                    debate_response += "\n\n" + "\n\n".join(tool_results)
            
            # Ensure we have some response content
            if not debate_response.strip():
                debate_response = f"[No response content from {cfg['name']}]"
            
            return {
                'response': debate_response.strip(),
                'agent_config': cfg,
                'agent_index': agent_index,
                'raw_response': response_obj
            }
            
        except Exception as e:
            # Handle any unexpected errors in response generation
            self.log(f"Error generating response for {cfg['name']}: {str(e)}", Colors.RED)
            error_response = f"Error generating response: {str(e)}"
            
            return {
                'response': error_response,
                'agent_config': cfg,
                'agent_index': agent_index,
                'raw_response': None
            }

    async def _generate_leader_question_async(self, question: str, conversation_history: List[str]) -> Dict:
        """Generate leader question asynchronously."""
        leader_system = self.prompts['leader_prompt'].format(question=question, history="\n".join(conversation_history))
        
        api_params = {
            "model": self.leader_model,
            "messages": [
                {"role": "system", "content": leader_system},
                {"role": "user", "content": self.prompts['user_messages']['leader_next_question']}
            ],
            "temperature": self.leader_temperature,
            "max_tokens": self.max_tokens_per_response,
            "top_p": self.top_p
        }
        
        response_obj = await self._generate_response_async(api_params)
        
        # Track API call costs
        if hasattr(response_obj, 'usage') and response_obj.usage:
            self._track_api_call(
                self.leader_model,
                response_obj.usage.prompt_tokens,
                response_obj.usage.completion_tokens,
                response_obj
            )
        
        leader_response = response_obj.choices[0].message.content.strip()
        
        return {
            'response': leader_response,
            'raw_response': response_obj
        }

    async def _generate_oracle_synthesis_async(self, question: str, conversation_history: List[str]) -> Dict:
        """Generate oracle synthesis asynchronously."""
        full_history = "\n".join(conversation_history)
        
        api_params = {
            "model": self.oracle_model,
            "messages": [
                {"role": "system", "content": self.prompts['oracle_prompt'].format(question=question, full_history=full_history)}
            ],
            "temperature": self.oracle_temperature,
            "max_tokens": self.max_tokens_final_response,
            "top_p": 1.0
        }
        
        response_obj = await self._generate_response_async(api_params)
        
        # Track Oracle API call costs
        if hasattr(response_obj, 'usage') and response_obj.usage:
            self._track_api_call(
                self.oracle_model,
                response_obj.usage.prompt_tokens,
                response_obj.usage.completion_tokens,
                response_obj
            )
        
        oracle_response = response_obj.choices[0].message.content.strip()
        
        return {
            'response': oracle_response,
            'raw_response': response_obj
        }

    def _calculate_api_cost(self, response_obj) -> Dict:
        """Calculate the cost of an API call using openai_cost_calculator."""
        try:
            cost_details = estimate_cost(response_obj)
            return {
                'input_cost': float(cost_details.get('prompt_cost_uncached', 0.0)),
                'output_cost': float(cost_details.get('completion_cost', 0.0)),
                'total_cost': float(cost_details.get('total_cost', 0.0))
            }
        except Exception as e:
            self.log(f"Warning: Could not calculate cost: {e}", Colors.YELLOW)
            return {'input_cost': 0.0, 'output_cost': 0.0, 'total_cost': 0.0}

    def _track_api_call(self, model_name: str, input_tokens: int, output_tokens: int, response_obj=None):
        """Track API call statistics and costs."""
        self.debug_info['api_calls'] += 1
        self.debug_info['total_input_tokens'] += input_tokens
        self.debug_info['total_output_tokens'] += output_tokens
        
        # Calculate cost for this call using the response object if available
        if response_obj:
            cost_details = self._calculate_api_cost(response_obj)
        else:
            # Fallback for when response object is not available
            cost_details = {'input_cost': 0.0, 'output_cost': 0.0, 'total_cost': 0.0}
            
        call_cost = cost_details['total_cost']
        self.debug_info['total_cost'] += call_cost
        
        # Track cost breakdown by model
        if model_name not in self.debug_info['cost_breakdown']:
            self.debug_info['cost_breakdown'][model_name] = {
                'calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0
            }
        
        self.debug_info['cost_breakdown'][model_name]['calls'] += 1
        self.debug_info['cost_breakdown'][model_name]['input_tokens'] += input_tokens
        self.debug_info['cost_breakdown'][model_name]['output_tokens'] += output_tokens
        self.debug_info['cost_breakdown'][model_name]['cost'] += call_cost

    def _track_agreement(self, is_agreement: bool):
        """Track agreement statistics."""
        self.debug_info['total_decisions'] += 1
        if is_agreement:
            self.debug_info['total_agreements'] += 1
        
        # Update agreement rate
        if self.debug_info['total_decisions'] > 0:
            self.debug_info['agreement_rate'] = (
                self.debug_info['total_agreements'] / self.debug_info['total_decisions']
            )

    def get_debug_summary(self) -> Dict:
        """Return a summary of debugging information."""
        return {
            'execution_time_seconds': self.debug_info['total_time'],
            'agreement_rate': self.debug_info['agreement_rate'],
            'total_agreements': self.debug_info['total_agreements'],
            'total_decisions': self.debug_info['total_decisions'],
            'total_api_calls': self.debug_info['api_calls'],
            'total_tokens': {
                'input': self.debug_info['total_input_tokens'],
                'output': self.debug_info['total_output_tokens'],
                'total': self.debug_info['total_input_tokens'] + self.debug_info['total_output_tokens']
            },
            'total_cost_usd': self.debug_info['total_cost'],
            'cost_breakdown_by_model': self.debug_info['cost_breakdown']
        }

    def print_debug_summary(self):
        """Print a formatted debug summary."""
        summary = self.get_debug_summary()
        
        print(f"\n{Colors.CYAN}=== Chain of Debate Debug Summary ==={Colors.RESET}")
        print(f"{Colors.WHITE}Execution Time: {summary['execution_time_seconds']:.2f} seconds{Colors.RESET}")
        print(f"{Colors.GREEN}Agreement Rate: {summary['agreement_rate']:.2%} ({summary['total_agreements']}/{summary['total_decisions']}){Colors.RESET}")
        print(f"{Colors.BLUE}Total API Calls: {summary['total_api_calls']}{Colors.RESET}")
        print(f"{Colors.BLUE}Total Tokens: {summary['total_tokens']['total']:,} (Input: {summary['total_tokens']['input']:,}, Output: {summary['total_tokens']['output']:,}){Colors.RESET}")
        print(f"{Colors.YELLOW}Total Cost: ${summary['total_cost_usd']:.4f}{Colors.RESET}")
        
        if summary['cost_breakdown_by_model']:
            print(f"\n{Colors.CYAN}Cost Breakdown by Model:{Colors.RESET}")
            for model, details in summary['cost_breakdown_by_model'].items():
                print(f"  {Colors.WHITE}{model}: ${details['cost']:.4f} ({details['calls']} calls, {details['input_tokens']+details['output_tokens']:,} tokens){Colors.RESET}")

    def get_log_file_path(self, question: str = None) -> str:
        """Generate the expected log file path for a given question (or current datetime)."""
        if not self.save_log:
            return None
        
        logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chain_of_debate_{timestamp}.txt"
        return os.path.join(logs_dir, filename)

    def log(self, message: str, color: str = Colors.WHITE):
        if self.verbose:
            print(f"{color}{message}{Colors.RESET}")
        
        # Add to process log if save_log is enabled
        if self.save_log:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
            self.process_log.append(f"[{timestamp}] {message}")

    def _save_process_log(self, question: str, final_answer: str):
        """Save the complete process log to a file with datetime-based filename."""
        if not self.save_log:
            return
        
        try:
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            # Generate filename with current datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chain_of_debate_{timestamp}.txt"
            filepath = os.path.join(logs_dir, filename)
            
            # Prepare the complete log content
            header = f"""Chain of Debate Session Log
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Question: {question}

Configuration:
- Debate Agents: {self.n_debate_agents}
- Max Rounds per Debate: {self.max_rounds_per_debate}
- Max Questions: {self.max_questions}
- Leader Model: {self.leader_model}
- Oracle Model: {self.oracle_model}
- Debate Model: {self.debate_model}

{"="*80}
PROCESS LOG:
{"="*80}

"""
            
            # Get debug summary for footer
            debug_summary = self.get_debug_summary()
            footer = f"""

{"="*80}
FINAL ANSWER:
{"="*80}
{final_answer}

{"="*80}
DEBUG SUMMARY:
{"="*80}
Execution Time: {debug_summary['execution_time_seconds']:.2f} seconds
Agreement Rate: {debug_summary['agreement_rate']:.2%} ({debug_summary['total_agreements']}/{debug_summary['total_decisions']})
Total API Calls: {debug_summary['total_api_calls']}
Total Tokens: {debug_summary['total_tokens']['total']:,} (Input: {debug_summary['total_tokens']['input']:,}, Output: {debug_summary['total_tokens']['output']:,})
Total Cost: ${debug_summary['total_cost_usd']:.4f}

Cost Breakdown by Model:
"""
            
            if debug_summary['cost_breakdown_by_model']:
                for model, details in debug_summary['cost_breakdown_by_model'].items():
                    footer += f"  {model}: ${details['cost']:.4f} ({details['calls']} calls, {details['input_tokens']+details['output_tokens']:,} tokens)\n"
            
            # Write the complete log to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write('\n'.join(self.process_log))
                f.write(footer)
            
            if self.verbose:
                print(f"{Colors.CYAN}Process log saved to: {filepath}{Colors.RESET}")
            
            self.last_log_file = filepath  # Store the filepath
            return filepath
            
        except Exception as e:
            if self.verbose:
                print(f"{Colors.RED}Error saving process log: {e}{Colors.RESET}")
            return None

    def run(self, question: str) -> str:
        """Run the Chain of Debate process (with optional async optimization)."""
        if self.use_async:
            return asyncio.run(self._run_async(question))
        else:
            # Fallback to synchronous implementation (not implemented here)
            # You would need to keep the original sync methods for this
            raise NotImplementedError("Synchronous mode not implemented. Use use_async=True.")

    async def _run_async(self, question: str) -> str:
        # Start timing
        start_time = time.time()
        
        self.log(f"Starting CoD for question: {question}", Colors.BLUE)

        # Initialize conversation history
        conversation_history = [f"Main Question: {question}"]
        done = False
        question_num = 0

        # Initialize main progress bar
        main_pbar = None
        if self.progressbar:
            # Estimate total steps: max_questions + 1 (for oracle)
            total_steps = self.max_questions + 1
            main_pbar = tqdm(total=total_steps, desc="Chain of Debate Progress", 
                           unit="step", colour="cyan", disable=not self.progressbar)

        while not done and question_num < self.max_questions:
            question_num += 1
            self.log(f"Leader generating question {question_num}", Colors.CYAN)

            # Leader generates next question
            leader_data = await self._generate_leader_question_async(question, conversation_history)
            leader_response = leader_data['response']

            self.log(f"Leader Question: {leader_response}", Colors.CYAN)

            # Check for [DONE]
            if "[DONE]" in leader_response:
                done = True
                leader_response = leader_response.split("[DONE]")[0].strip()
                if leader_response:
                    conversation_history.append(f"Leader Question {question_num}: {leader_response}")
                if main_pbar:
                    main_pbar.update(1)
                    main_pbar.set_description(f"Leader finished (Question {question_num})")
                break

            current_question = leader_response
            conversation_history.append(f"Leader Question {question_num}: {current_question}")
            
            if main_pbar:
                main_pbar.set_description(f"Processing Question {question_num}: Debate Phase")

            # Debate agents debate
            prior_context = '\n'.join(conversation_history[:-1])
            debate_history = [f"Answering: {current_question}\nPrior Context: {prior_context}"]
            agreed_on_answer = False
            round_num = 0
            is_reflection_round = False

            # Initialize debate progress bar
            debate_pbar = None
            if self.progressbar:
                total_debate_steps = self.max_rounds_per_debate * len(self.debate_agent_configs)
                debate_pbar = tqdm(total=total_debate_steps, desc=f"Debate for Q{question_num}", 
                                 unit="agent", leave=False, colour="green", disable=not self.progressbar)

            while not agreed_on_answer and round_num < self.max_rounds_per_debate:
                round_num += 1
                self.log(f"Debate Round {round_num} (Reflection: {is_reflection_round})", Colors.WHITE)
                agreements = []
                all_refined_texts = []

                is_reflection_round = not is_reflection_round

                # Log reflection round status
                if is_reflection_round:
                    self.log("Reflection Round Active", Colors.YELLOW)

                # Generate all debate responses asynchronously
                self.log("Generating debate responses asynchronously...", Colors.BLUE)
                response_data_list = await self._generate_debate_responses_async(
                    self.debate_agent_configs, current_question, question, 
                    debate_history, is_reflection_round
                )

                # Process responses sequentially to maintain history order
                for response_data in response_data_list:
                    agent_name = response_data['agent_config']['name']
                    debate_response = response_data['response']
                    cfg = response_data['agent_config']
                    
                    self.log(f"{agent_name} Response: {debate_response}", Colors.GREEN)

                    # Append to history
                    debate_history.append(f"{cfg['name']}: {debate_response}")

                    # Parse tag and track agreements
                    last_line = debate_response.splitlines()[-1].strip() if debate_response.splitlines() else ""
                    
                    # Handle cases where response might be empty or malformed
                    if not last_line:
                        self.log(f"Warning: Empty response from {agent_name}", Colors.YELLOW)
                        agreements.append((False, "Empty response"))
                        self._track_agreement(False)
                        self.log(f"Invalid response from {agent_name}: Empty", Colors.RED)
                    elif last_line.startswith("[AGREE:"):
                        try:
                            refined_text = last_line.split("[AGREE:", 1)[1].rstrip("]").strip()
                            if refined_text:
                                agreements.append((True, refined_text))
                                all_refined_texts.append(refined_text)
                                self._track_agreement(True)  # Track agreement
                                self.log(f"Agreement from {agent_name}: {refined_text}", Colors.GREEN)
                            else:
                                self.log(f"Warning: Empty AGREE tag from {agent_name}", Colors.YELLOW)
                                agreements.append((False, "Empty AGREE tag"))
                                self._track_agreement(False)
                        except Exception as e:
                            self.log(f"Warning: Malformed AGREE tag from {agent_name}: {e}", Colors.YELLOW)
                            agreements.append((False, f"Malformed AGREE tag: {str(e)}"))
                            self._track_agreement(False)
                    elif last_line.startswith("[DISAGREE:"):
                        try:
                            disagree_text = last_line.split("[DISAGREE:", 1)[1].rstrip("]").strip()
                            if disagree_text:
                                agreements.append((False, disagree_text))
                                self._track_agreement(False)  # Track disagreement
                                self.log(f"Disagreement from {agent_name}: {disagree_text}", Colors.RED)
                            else:
                                self.log(f"Warning: Empty DISAGREE tag from {agent_name}", Colors.YELLOW)
                                agreements.append((False, "Empty DISAGREE tag"))
                                self._track_agreement(False)
                        except Exception as e:
                            self.log(f"Warning: Malformed DISAGREE tag from {agent_name}: {e}", Colors.YELLOW)
                            agreements.append((False, f"Malformed DISAGREE tag: {str(e)}"))
                            self._track_agreement(False)
                    else:
                        agreements.append((False, "Invalid tag"))
                        self._track_agreement(False)  # Track invalid response as disagreement
                        self.log(f"Invalid tag from {agent_name}: {last_line[:50]}{'...' if len(last_line) > 50 else ''}", Colors.RED)
                    
                    # Update debate progress bar
                    if debate_pbar:
                        status = "AGREE" if agreements[-1][0] else "DISAGREE"
                        debate_pbar.update(1)
                        debate_pbar.set_description(f"R{round_num} {agent_name[:15]}: {status}")

            # Check consensus
            if all(a[0] for a in agreements):
                base_text = all_refined_texts[0]
                if all(difflib.SequenceMatcher(None, base_text, t).ratio() > 0.9 for t in all_refined_texts):
                    agreed_on_answer = True
                    agreed_answer = base_text
                    self.log(f"Consensus Reached: {agreed_answer}", Colors.PURPLE)
                else:
                    self.log("Consensus Failed: Texts not similar", Colors.RED)

            if agreed_on_answer:
                conversation_history.append(f"Debate Team Consensus Answer: {agreed_answer}")
            else:
                fallback_answer = Counter([a[1] for a in agreements if a[0]]).most_common(1)[0][0] if any(a[0] for a in agreements) else "No consensus reached; partial insights: [summarize debate_history]"
                conversation_history.append(f"Debate Team Fallback Answer: {fallback_answer}")
                self.log(f"Fallback Answer: {fallback_answer}", Colors.PURPLE)
            
            # Close debate progress bar and update main progress
            if debate_pbar:
                debate_pbar.close()
            if main_pbar:
                main_pbar.update(1)
                status = "Consensus" if agreed_on_answer else "Fallback"
                main_pbar.set_description(f"Q{question_num} Complete ({status})")

        # Oracle Synthesis
        self.log("Oracle Synthesizing Final Answer", Colors.WHITE)
        if main_pbar:
            main_pbar.set_description("Oracle Synthesis")
            
        oracle_data = await self._generate_oracle_synthesis_async(question, conversation_history)
        oracle_response = oracle_data['response']

        # Calculate total execution time
        end_time = time.time()
        self.debug_info['total_time'] = end_time - start_time

        # Complete progress bar
        if main_pbar:
            main_pbar.update(1)
            main_pbar.set_description("Chain of Debate Complete")
            main_pbar.close()

        self.log(f"Final Answer: {oracle_response}", Colors.PURPLE)
        
        # Save process log to file
        if self.save_log:
            self._save_process_log(question, oracle_response)
        
        # Print debug summary if verbose
        if self.verbose:
            self.print_debug_summary()

        return oracle_response
