# edo/__main__.py

"""
Entry point for the edo package.
Run with: python -m edo
"""

import sys
import shutil
import argparse
from pyodys import list_schemes

# ANSI color codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

def center_lines(text):
    """Centers each line of a multi-line string based on terminal width."""
    terminal_width = shutil.get_terminal_size().columns
    centered_lines = [line.strip().center(terminal_width) for line in text.split('\n')]
    return '\n'.join(centered_lines)

def print_available_schemes():
    """
    Prints a list of all available Runge-Kutta schemes.
    """
    print("Available Time Integrator Schemes:")
    for name in sorted(list_schemes):
        print(f"- {name}")

def main():
    parser = argparse.ArgumentParser(description="Solve ODEs using various Runge-Kutta methods.")
    parser.add_argument('--list-schemes', action='store_true', help="List all available Runge-Kutta schemes.")
    #parser.add_argument('--help', action='store_true', help="List all available Runge-Kutta schemes.")
    
    args = parser.parse_args()
    
    if args.list_schemes:
        print_available_schemes()
        return

    banner = f"""
        {RED}███████╗{GREEN}██████╗   {YELLOW}██████╗
        {RED}██╔════╝{GREEN}██╔═══██ {YELLOW}██╔═══██╗
        {RED}█████╗  {GREEN}██║   ██ {YELLOW}██║   ██║
        {RED}██╔══╝  {GREEN}██║   ██ {YELLOW}██║   ██║
        {RED}███████╗{GREEN}██████╔╝ {YELLOW}╚██████╔╝
        {RED}╚══════╝{GREEN}╚═════╝   {YELLOW}╚═════╝
        {CYAN}Numerical EDO Solver {RESET}
    """
    print(center_lines(banner))
    # Center other text
    separator = "=" * 60
    welcome_message = f"{MAGENTA}Welcome to the EDO solver package{RESET}"
    
    print(center_lines(separator))
    print(center_lines(welcome_message))
    print(center_lines(separator))
    
    print("\nAvailable demos:")
    print("  1. Van der Pol Oscillator")
    print("  2. Lorenz System")
    print("  3. Robertson Model")
    print("  4. HIRES System")
    print("  5. etc.")
    print(f"\n{BLUE}Enjoy solving ODEProblem! {RESET}")

    
    



if __name__ == "__main__":
    sys.exit(main())