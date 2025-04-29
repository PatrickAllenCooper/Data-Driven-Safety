#!/usr/bin/env python3
"""
Run all examples for the data-driven safety verification project.
"""

import os
import sys
import argparse
import time

def run_example(example_name):
    """
    Run a specific example.
    
    Args:
        example_name (str): Name of the example to run.
        
    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    example_path = os.path.join("examples", f"{example_name}.py")
    
    if not os.path.exists(example_path):
        print(f"Error: Example '{example_name}' not found at {example_path}")
        return 1
    
    print(f"Running example: {example_name}")
    print("=" * 80)
    
    start_time = time.time()
    return_code = os.system(f"{sys.executable} {example_path}")
    end_time = time.time()
    
    print("=" * 80)
    print(f"Example completed in {end_time - start_time:.2f} seconds")
    print(f"Return code: {return_code}")
    print()
    
    return return_code

def main():
    """
    Run all examples or a specific example.
    """
    parser = argparse.ArgumentParser(description="Run examples for the data-driven safety verification project.")
    parser.add_argument(
        "example", nargs="?", choices=["room_temperature", "comparison", "all"],
        default="all", help="Example to run (default: all)"
    )
    args = parser.parse_args()
    
    if args.example == "all":
        examples = ["room_temperature", "comparison"]
    else:
        examples = [args.example]
    
    success = True
    for example in examples:
        result = run_example(example)
        if result != 0:
            success = False
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 