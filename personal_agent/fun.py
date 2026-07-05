"""
Demo multi-turn conversation with SQLite checkpointer.

Chay 2 luot chat tren cung session_id de kiem tra
agent co nho ngu canh hoi thoai khong.
"""

from agent.graph import invoke_agent
from agent.profiles import setup_profiles
from tools.registry import setup_tools

# Setup tools va profiles
setup_tools()
setup_profiles(yaml_path="config/profiles.yaml")

# Session ID co dinh de test multi-turn
SESSION_ID = "demo-session-004"

print("=" * 60)
print("DEMO: Multi-turn Conversation with SQLite Checkpointer")
print("=" * 60)

# -- Luot 1: Hoi cau hoi --
print("\n Hãy nhớ tên của tôi là Lương Anh Minh")
res1 = invoke_agent(
    query="Hãy nhớ tên của tôi là Lương Anh Minh",
    session_id=SESSION_ID,
    max_steps=5,
)
print(f"[Turn 1] Agent: {res1.get('final_answer')}")
print(f"         Steps: {res1.get('current_step')}")
print(f"         Status: {res1.get('status')}")

# -- Luot 2: Hoi tiep -- agent nen nho ngu canh --
print("\n" + "-" * 60)
print("\n dựa theo những gì tôi đã nói , Tên của tôi là gì ? ")
res2 = invoke_agent(
    query="Tên của tôi là gì ?",
    session_id=SESSION_ID,
    max_steps=5,
)
print(f"[Turn 2] Agent: {res2.get('final_answer')}")
print(f"         Steps: {res2.get('current_step')}")
print(f"         Status: {res2.get('status')}")
print(f"         message: {res2.get('messages')}")


print("\n" + "=" * 60)
print("DONE -- Check file ./data/checkpoints.sqlite")
print("=" * 60)
