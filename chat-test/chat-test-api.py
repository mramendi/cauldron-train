import time
import sys
import argparse
from openai import OpenAI
from pathlib import Path

# Global variables set by argument parsing
client = None
conversation_history = []
show_metrics = False
temperature = 0.6  # Default temperature, can be overridden by script
model_name = None  # Set by argument parsing


def load_script(script_path):
    """
    Load a script file and extract system prompt + user messages.

    Format:
    - First line: "0" = no system prompt, temperature=0 (deterministic mode)
    - First line: system prompt (empty/whitespace = no system prompt, temperature=0.6)
    - Subsequent non-empty lines: user messages

    Returns: (system_prompt, user_messages, temperature)
    """
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into lines, handling different line endings
    lines = content.splitlines()

    if not lines:
        return None, [], 0.6

    # Check if first line is "0" for deterministic mode
    first_line = lines[0].strip()
    if first_line == "0":
        # Deterministic mode: no system prompt, temperature=0
        user_messages = [line.strip() for line in lines[1:] if line.strip()]
        return None, user_messages, 0.0

    # Normal mode: first line is system prompt (may be empty)
    system_prompt = first_line if first_line else None

    # Remaining non-empty lines are user messages
    user_messages = [line.strip() for line in lines[1:] if line.strip()]

    return system_prompt, user_messages, 0.6

def send_message(user_input):
    """Send a message and get response, updating conversation history."""
    global conversation_history, temperature, model_name

    # Append User Message to History
    conversation_history.append({"role": "user", "content": user_input})

    print("Assistant: ", end="", flush=True)

    # Start Timing
    start_time = time.perf_counter()
    first_token_time = None
    last_token_time = None
    token_count = 0
    full_response_content = ""
    in_reasoning = False

    # Stream the Response
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=conversation_history,
            stream=True,
            temperature=temperature,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        for chunk in stream:
            # Skip chunks with no choices (WandB API quirk)
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Check for reasoning content (o1/GLM-style thinking)
            reasoning_delta = None
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_delta = delta.reasoning_content
            elif hasattr(delta, 'reasoning') and delta.reasoning:
                reasoning_delta = delta.reasoning

            # Display reasoning if present
            if reasoning_delta:
                if not in_reasoning:
                    print("\n(reasoning) ", end="", flush=True)
                    in_reasoning = True
                print(reasoning_delta, end="", flush=True)
                continue

            # Handle regular content
            content_delta = delta.content if hasattr(delta, 'content') else None

            if content_delta:
                # If we were in reasoning mode, add newline before content
                if in_reasoning:
                    print("\nAssistant: ", end="", flush=True)
                    in_reasoning = False

                current_time = time.perf_counter()

                # Capture TTFT on the very first token
                if first_token_time is None:
                    first_token_time = current_time

                last_token_time = current_time
                token_count += 1
                full_response_content += content_delta

                # Print token immediately
                print(content_delta, end="", flush=True)

    except Exception as e:
        print(f"\nError calling API: {e}")
        return False

    print()  # Newline after response

    # Calculate and display metrics if enabled
    if show_metrics:
        print("\n--- Performance Metrics ---")

        if first_token_time:
            # TTFT: Time from Request Start -> First Token
            ttft_ms = (first_token_time - start_time) * 1000
            print(f"Latency (TTFT): {ttft_ms:.2f} ms")

            # TPS: (Total Tokens - 1) / (Time form 1st to Last token)
            if token_count > 1:
                generation_time = last_token_time - first_token_time
                tps = (token_count - 1) / generation_time
                print(f"Throughput:     {tps:.2f} tokens/sec")

            print(f"Total Tokens:   {token_count}")
        else:
            print("No tokens generated.")

    # Append Assistant Response to History
    conversation_history.append({"role": "assistant", "content": full_response_content})

    return True


def run_script(user_messages):
    """Execute scripted user messages sequentially."""
    print("--- Running script ---\n")

    for user_input in user_messages:
        print(f"User: {user_input}")

        if not send_message(user_input):
            print("Error during script execution, stopping.")
            break

        print()  # Blank line between exchanges

    print("--- Script complete ---\n")


def chat_loop(script_messages=None):
    """Main chat loop with optional script execution."""
    print("--- API Chat Client (Type 'quit' to exit) ---")
    print(f"Connected to: {client.base_url}")
    print(f"Model: {model_name}\n")

    # If script provided, run it first
    if script_messages:
        run_script(script_messages)
        print("Continuing in interactive mode with script context preserved...\n")

    while True:
        # Get User Input
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting...")
                break
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        # Send message and get response
        send_message(user_input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive chat client for OpenAI-compatible API"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Base URL for API (e.g., http://localhost:8080/v1)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="sk-no-key-required",
        help="API key (default: sk-no-key-required)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use"
    )
    parser.add_argument(
        "--script",
        type=str,
        help="Script file to execute before interactive mode (optional)"
    )
    parser.add_argument(
        "--tps",
        action="store_true",
        help="Show TTFT/TPS performance metrics (default: hidden)"
    )

    args = parser.parse_args()

    # Set global flags
    show_metrics = args.tps
    model_name = args.model

    # Initialize client
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key
    )

    # Load script if provided
    script_messages = None
    if args.script:
        system_prompt, user_messages, script_temperature = load_script(args.script)

        # Set global temperature from script
        temperature = script_temperature

        # Initialize conversation history with system prompt if present
        if system_prompt:
            conversation_history = [{"role": "system", "content": system_prompt}]
            print(f"System prompt: {system_prompt}\n")
        else:
            conversation_history = []
            if script_temperature == 0.0:
                print("Deterministic mode: temperature=0, no system prompt\n")
            else:
                print("No system prompt (empty first line)\n")

        # Show what was loaded
        if user_messages:
            print(f"Loaded {len(user_messages)} user message(s) from script:\n")
            for i, msg in enumerate(user_messages, 1):
                preview = msg[:50] + "..." if len(msg) > 50 else msg
                print(f"  {i}. {preview}")
            print()
        else:
            print("Warning: No user messages found in script file!\n")

        script_messages = user_messages
    else:
        # No script, start with empty history (tokenizer will add default)
        conversation_history = []

    # Start chat loop
    chat_loop(script_messages)
