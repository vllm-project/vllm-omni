import re

from vllm.logger import current_formatter_type, init_logger

logger = init_logger(__name__)

ORANGE = "\033[38;5;208m"
BLUE = "\033[34m"
WHITE = "\033[97m"
PURPLE = "\033[35m"
RESET = "\033[0m"

# Official vLLM Logo Base
VLLM_L1 = f"       {WHITE}█     █     █▄   ▄█{RESET}"
VLLM_L2 = f" {ORANGE}▄▄{RESET} {BLUE}▄█{RESET} {WHITE}█     █     █ ▀▄▀ █{RESET}"
VLLM_L3 = f"  {ORANGE}█{RESET}{BLUE}▄█▀{RESET} {WHITE}█     █     █     █{RESET}"
VLLM_L4 = f"   {BLUE}▀▀{RESET}  {WHITE}▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀{RESET}"

GAP_L1 = "    "
GAP_L2 = f"{WHITE}  ▄▄▄ {RESET}"
GAP_L3 = f"{WHITE}      {RESET}"
GAP_L4 = "      "

O_L1 = f"{BLUE}   ▄▀▀{ORANGE}▀▀▄ {RESET}"
O_L2 = f"{BLUE} █    {ORANGE}█ {RESET}"
O_L3 = f"{BLUE} █    {ORANGE}█ {RESET}"
O_L4 = f"{PURPLE}  ▀▀▀▀  {RESET}"
MNI_L1 = f"{WHITE}█▄   ▄█ █▄    █ ▀█▀ {RESET}"
MNI_L2 = f"{WHITE}█ ▀▄▀ █ █ ▀▄  █  █  {RESET}"
MNI_L3 = f"{WHITE}█     █ █   ▀▄█  █  {RESET}"
MNI_L4 = f"{WHITE}▀     ▀ ▀     ▀ ▀▀▀ {RESET}"

LOGO = f"""{VLLM_L1}{GAP_L1}{O_L1}{MNI_L1}
{VLLM_L2}{GAP_L2}{O_L2}{MNI_L2}
{VLLM_L3}{GAP_L3}{O_L3}{MNI_L3}
{VLLM_L4}{GAP_L4}{O_L4}{MNI_L4}
"""


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def log_logo() -> None:
    logo = LOGO if current_formatter_type(logger) == "color" else _ANSI_RE.sub("", LOGO)
    logger.info(logo)
