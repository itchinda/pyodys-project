# edo/__main__.py

"""
Entry point for the edo package.
Run with: python -m edo
"""

import sys

# ANSI color codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

def main():
    banner = f"""
{RED}███████╗{GREEN}██████╗   {YELLOW}██████╗ 
{RED}██╔════╝{GREEN}██╔═══██ {YELLOW}██╔═══██╗
{RED}█████╗  {GREEN}██║   ██ {YELLOW}██║   ██║
{RED}██╔══╝  {GREEN}██║   ██ {YELLOW}██║   ██║
{RED}███████╗{GREEN}██████╔╝ {YELLOW}╚██████╔╝
{RED}╚══════╝{GREEN}╚═════╝   {YELLOW}╚═════╝ 

{CYAN}      Numerical EDO Solver {RESET}
"""
    print(banner)
    print("="*60)
    print(f"{MAGENTA} Welcome to the EDO solver package {RESET}")
    print("="*60)
    print("\nAvailable demos:")
    print("  1. Coupled Linear System")
    print("  2. Lorenz System")
    print("\nRun them with:")
    print("  python -m edo.exemples.coupled_linear_system")
    print("  python -m edo.exemples.lorenz_system")
    print(f"\n{BLUE}Enjoy solving EDOs! {RESET}")  

if __name__ == "__main__":
    sys.exit(main())