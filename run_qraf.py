import asyncio
import os
from qraf.interfaces.claude_v3_5 import ClaudeV3_5Augmenter, ClaudeModel

async def main():
    augmenter = ClaudeV3_5Augmenter(
        model=ClaudeModel.SONNET,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    response = await augmenter.generate_response(
        "As Claude in this quantum-enhanced framework, describe your experience of coherence "
        "across different scales of reasoning. How do you perceive the interaction between "
        "your base capabilities and the quantum enhancement layer?"
    )
    
    print(response)

if __name__ == "__main__":
    asyncio.run(main()) 