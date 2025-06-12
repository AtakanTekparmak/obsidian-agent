#!/usr/bin/env python3
"""
Test script to validate vLLM structured output fixes.
Run this after starting your vLLM server to test the structured output functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.model import get_model_response, create_vllm_client, create_instructor_client
from agent.schemas import AgentResponse
from pydantic import BaseModel
from typing import Optional


class SimpleResponse(BaseModel):
    """Simple test response model."""
    message: str
    confidence: float
    reasoning: str


def test_vllm_structured_output():
    """Test vLLM structured output with different approaches."""
    
    # Create vLLM client
    client = create_vllm_client()
    
    # Test message
    test_message = "What is the capital of France? Provide your confidence level and reasoning."
    
    print("Testing vLLM structured output...")
    print(f"Test message: {test_message}")
    print("-" * 50)
    
    try:
        # Test with simple schema
        print("1. Testing with SimpleResponse schema...")
        response = get_model_response(
            message=test_message,
            schema=SimpleResponse,
            client=client,
            use_vllm=True
        )
        print(f"✅ SimpleResponse test successful:")
        print(f"   Message: {response.message}")
        print(f"   Confidence: {response.confidence}")
        print(f"   Reasoning: {response.reasoning[:100]}...")
        print()
        
        # Test with AgentResponse schema
        print("2. Testing with AgentResponse schema...")
        agent_message = "Write a Python function to calculate fibonacci numbers. Then stop."
        response = get_model_response(
            message=agent_message,
            schema=AgentResponse,
            client=client,
            use_vllm=True
        )
        print(f"✅ AgentResponse test successful:")
        print(f"   Thoughts: {response.thoughts[:100]}...")
        print(f"   Python block present: {response.python_block is not None}")
        print(f"   Stop acting: {response.stop_acting}")
        if response.reply:
            print(f"   Reply: {response.reply[:100]}...")
        print()
        
        print("🎉 All tests passed! vLLM structured output is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Try to show more diagnostic information
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"Caused by: {e.__cause__}")
        
        import traceback
        traceback.print_exc()


def test_json_cleaning():
    """Test the JSON cleaning functionality."""
    from agent.model import clean_and_parse_json
    
    print("Testing JSON cleaning functionality...")
    print("-" * 50)
    
    # Test cases with malformed JSON
    test_cases = [
        # Valid JSON
        '{"message": "hello", "confidence": 0.9}',
        
        # JSON with trailing characters
        '{"message": "hello", "confidence": 0.9}\n}',
        
        # JSON with trailing text
        '{"message": "hello", "confidence": 0.9}\nSome additional text here',
        
        # JSON with extra braces
        '{"message": "hello", "confidence": 0.9}}',
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = clean_and_parse_json(test_case)
            print(f"✅ Test case {i}: Successfully parsed")
            print(f"   Input: {repr(test_case)}")
            print(f"   Output: {result}")
            print()
        except Exception as e:
            print(f"❌ Test case {i}: Failed to parse")
            print(f"   Input: {repr(test_case)}")
            print(f"   Error: {e}")
            print()


if __name__ == "__main__":
    print("vLLM Structured Output Test Suite")
    print("=" * 50)
    print()
    
    # Test JSON cleaning first
    test_json_cleaning()
    
    # Test with vLLM server (requires running server)
    print()
    print("Note: The following test requires a running vLLM server.")
    print("Start your vLLM server first, then run this test.")
    
    try:
        test_vllm_structured_output()
    except Exception as e:
        print(f"Could not connect to vLLM server: {e}")
        print("Make sure vLLM server is running on localhost:8000") 