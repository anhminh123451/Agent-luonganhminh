# from agent.profiles import setup_profiles , list_profiles,get_profile
# from tools.registry import setup_tools
# from agent.prompts import build_system_prompt_for_profile,build_system_prompt
# from agent.state import AgentState

# setup_tools()
# setup_profiles(yaml_path="config/profiles.yaml")

from agent.runner import _extract_json_from_text

text = "     xin chào bạn     "


print(text.strip())