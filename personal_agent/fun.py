from agent.prompts import  build_system_prompt_for_profile
from tools.registry import ToolRegistry,setup_tools

setup_tools()
available_tool = ToolRegistry.get_tools_for_profile("agent_core")


prompt = build_system_prompt_for_profile(profile_name="agent_core", agent_name="AI Assistant")

print(prompt)

                                    