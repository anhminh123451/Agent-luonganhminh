from agent.profiles import setup_profiles , list_profiles
from tools.registry import setup_tools
from agent.prompts import build_system_prompt_for_profile,build_system_prompt

setup_tools()
setup_profiles(yaml_path="config/profiles.yaml")

prompt = build_system_prompt_for_profile(profile_name="general_agent")
print(prompt)



                                    

                                    