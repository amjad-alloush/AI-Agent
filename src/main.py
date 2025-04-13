import os
import json
import argparse
from venv import load_dotenv
from agent import Agent


def main():
    """Main entry point for the AI agent CLI."""

    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Agent CLI")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="agent_state",
        help="Directory to save/load agent state",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt for the agent",
    )
    args = parser.parse_args()

    # Create agent
    agent = Agent(system_prompt=args.system_prompt)

    # Load state if available
    if os.path.exists(args.save_dir):
        print(f"Loading agent state from {args.save_dir}...")
        agent.load_state(args.save_dir)

    print("AI Agent initialized. Type 'exit' to quit, 'save' to save state.")
    print("=" * 50)

    # Main interaction loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "save":
            agent.save_state(args.save_dir)
            print(f"Agent state saved to {args.save_dir}")
            continue

    # Process user input
    input_data = {"type": "text", "content": user_input, "metadata": {"source": "cli"}}
    response = agent.process_input(input_data)

    # Display response
    print("\nAgent:", response["response"])
    if __name__ == "__main__":
        main()