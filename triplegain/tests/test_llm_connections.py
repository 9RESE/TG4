#!/usr/bin/env python3
"""
Test LLM API Connections

Quick test to verify all LLM APIs are reachable and working.
"""

import asyncio
import os
import sys
from pathlib import Path

# Load environment
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT))

from triplegain.src.llm.clients import OllamaClient, DeepSeekClient, AnthropicClient, OpenAIClient, XAIClient


async def test_ollama():
    """Test Ollama connection."""
    print("\n[1/5] Testing Ollama (qwen2.5:7b)...")
    try:
        client = OllamaClient({
            "base_url": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            "timeout_seconds": 60,
            "json_mode": False,  # Disable JSON mode for simple test
        })
        response = await client.generate(
            model="qwen2.5:7b",  # Use available model
            system_prompt="You are a helpful assistant. Respond briefly.",
            user_message="Say 'Hello' in one word.",
        )
        print(f"  ✅ Ollama OK: {response.text[:50]}...")
        return True
    except Exception as e:
        print(f"  ❌ Ollama FAILED: {e}")
        return False


async def test_deepseek():
    """Test DeepSeek connection."""
    print("\n[2/5] Testing DeepSeek...")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("  ⚠️ DeepSeek: No API key configured")
        return False
    try:
        client = DeepSeekClient({
            "api_key": api_key,
            "timeout_seconds": 30,
            "json_mode": False,  # Disable JSON mode for simple test
        })
        response = await client.generate(
            model="deepseek-chat",
            system_prompt="You are a helpful assistant. Respond briefly.",
            user_message="Say 'Hello' in one word.",
        )
        print(f"  ✅ DeepSeek OK: {response.text[:50]}...")
        return True
    except Exception as e:
        print(f"  ❌ DeepSeek FAILED: {e}")
        return False


async def test_anthropic():
    """Test Anthropic (Claude) connection."""
    print("\n[3/5] Testing Anthropic (Claude)...")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("  ⚠️ Anthropic: No API key configured")
        return False
    try:
        client = AnthropicClient({
            "api_key": api_key,
            "timeout_seconds": 30,
            "json_mode": False,  # Disable JSON mode for simple test
        })
        response = await client.generate(
            model="claude-3-5-haiku-20241022",  # Try Haiku (cheaper, more accessible)
            system_prompt="You are a helpful assistant. Respond briefly.",
            user_message="Say 'Hello' in one word.",
        )
        print(f"  ✅ Anthropic OK: {response.text[:50]}...")
        return True
    except Exception as e:
        print(f"  ❌ Anthropic FAILED: {e}")
        return False


async def test_openai():
    """Test OpenAI (GPT) connection."""
    print("\n[4/5] Testing OpenAI (GPT)...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  ⚠️ OpenAI: No API key configured")
        return False
    try:
        client = OpenAIClient({
            "api_key": api_key,
            "timeout_seconds": 30,
            "json_mode": False,  # Disable JSON mode for simple test
        })
        response = await client.generate(
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant. Respond briefly.",
            user_message="Say 'Hello' in one word.",
        )
        print(f"  ✅ OpenAI OK: {response.text[:50]}...")
        return True
    except Exception as e:
        print(f"  ❌ OpenAI FAILED: {e}")
        return False


async def test_xai():
    """Test xAI (Grok) connection."""
    print("\n[5/5] Testing xAI (Grok)...")
    api_key = os.getenv("XAI_BEARER_API_KEY")
    if not api_key:
        print("  ⚠️ xAI: No API key configured")
        return False
    try:
        client = XAIClient({
            "api_key": api_key,
            "timeout_seconds": 30,
            "json_mode": False,  # Disable JSON mode for simple test
        })
        response = await client.generate(
            model="grok-3",  # Use grok-3 (grok-beta deprecated)
            system_prompt="You are a helpful assistant. Respond briefly.",
            user_message="Say 'Hello' in one word.",
        )
        print(f"  ✅ xAI OK: {response.text[:50]}...")
        return True
    except Exception as e:
        print(f"  ❌ xAI FAILED: {e}")
        return False


async def main():
    print("=" * 60)
    print("  LLM API Connection Test")
    print("=" * 60)

    results = {
        "Ollama": await test_ollama(),
        "DeepSeek": await test_deepseek(),
        "Anthropic": await test_anthropic(),
        "OpenAI": await test_openai(),
        "xAI": await test_xai(),
    }

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name}: {status}")

    print(f"\n  Total: {passed}/{total} LLM APIs working")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
