"""
Utility functions for Chain of Debate.

This module contains helper functions and utilities used throughout the
Chain of Debate system.
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime


def load_json_config(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON configuration file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data as a dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the JSON is invalid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}")


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Parsed YAML data as a dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the YAML is invalid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in file {file_path}: {e}")


def save_json_config(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON configuration file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the JSON file
        indent: JSON indentation level
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_yaml_config(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to a YAML configuration file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save the YAML file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def validate_agent_config(config: Dict[str, Any]) -> bool:
    """
    Validate an agent configuration dictionary.
    
    Args:
        config: Agent configuration to validate
        
    Returns:
        True if valid, raises ValueError if not
        
    Raises:
        ValueError: If the configuration is invalid
    """
    required_fields = ['name', 'description', 'temperature']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Agent config missing required field: {field}")
    
    if not isinstance(config['name'], str) or not config['name'].strip():
        raise ValueError("Agent name must be a non-empty string")
    
    if not isinstance(config['description'], str) or not config['description'].strip():
        raise ValueError("Agent description must be a non-empty string")
    
    if not isinstance(config['temperature'], (int, float)) or config['temperature'] < 0:
        raise ValueError("Agent temperature must be a non-negative number")
    
    return True


def validate_agent_configs(configs: List[Dict[str, Any]]) -> bool:
    """
    Validate a list of agent configurations.
    
    Args:
        configs: List of agent configurations to validate
        
    Returns:
        True if all are valid, raises ValueError if not
        
    Raises:
        ValueError: If any configuration is invalid
    """
    if not isinstance(configs, list) or len(configs) == 0:
        raise ValueError("Agent configs must be a non-empty list")
    
    names = set()
    for i, config in enumerate(configs):
        validate_agent_config(config)
        
        # Check for duplicate names
        if config['name'] in names:
            raise ValueError(f"Duplicate agent name at index {i}: {config['name']}")
        names.add(config['name'])
    
    return True


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format a timestamp for use in filenames and logs.
    
    Args:
        timestamp: Datetime object, or None for current time
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.strftime("%Y%m%d_%H%M%S")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string for use as a filename.
    
    Args:
        filename: Original filename string
        
    Returns:
        Sanitized filename safe for filesystem use
    """
    # Remove or replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 100:
        filename = filename[:97] + "..."
    
    # Remove leading/trailing whitespace and periods
    filename = filename.strip('. ')
    
    # Ensure it's not empty
    if not filename:
        filename = "untitled"
    
    return filename


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of the result
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def count_tokens_estimate(text: str) -> int:
    """
    Rough estimate of token count for a text string.
    
    This is a simple approximation - for exact counts, use the OpenAI
    tokenizer library.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    # Rough approximation: 4 characters per token on average
    return len(text) // 4 + 1


def parse_agree_disagree_response(response: str) -> tuple[bool, str]:
    """
    Parse [AGREE:...] or [DISAGREE:...] tags from agent responses.
    
    Args:
        response: Agent response text
        
    Returns:
        Tuple of (is_agreement, extracted_text)
    """
    lines = response.strip().splitlines()
    if not lines:
        return False, "Empty response"
    
    last_line = lines[-1].strip()
    
    if last_line.startswith("[AGREE:") and last_line.endswith("]"):
        text = last_line[7:-1].strip()  # Remove [AGREE: and ]
        return True, text
    elif last_line.startswith("[DISAGREE:") and last_line.endswith("]"):
        text = last_line[10:-1].strip()  # Remove [DISAGREE: and ]
        return False, text
    else:
        return False, "Invalid or missing agreement tag"


def extract_done_marker(response: str) -> tuple[str, bool]:
    """
    Extract [DONE] marker from leader responses.
    
    Args:
        response: Leader response text
        
    Returns:
        Tuple of (response_without_done, has_done_marker)
    """
    if "[DONE]" in response:
        cleaned_response = response.replace("[DONE]", "").strip()
        return cleaned_response, True
    else:
        return response.strip(), False


def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory to create
    """
    os.makedirs(directory_path, exist_ok=True)


def get_package_config_path(config_filename: str) -> str:
    """
    Get the full path to a configuration file in the package.
    
    Args:
        config_filename: Name of the config file
        
    Returns:
        Full path to the config file
    """
    package_dir = os.path.dirname(__file__)
    return os.path.join(package_dir, 'config', config_filename)
