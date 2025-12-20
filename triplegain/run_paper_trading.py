#!/usr/bin/env python3
"""
TripleGain Paper Trading Runner

Main entry point for running the paper trading system with:
- Live price feeds from database/WebSocket
- LLM-based agents (TA, Regime, Trading Decision)
- Paper trading execution (simulated trades)
- Full coordinator orchestration

Usage:
    python -m triplegain.run_paper_trading

Environment:
    Requires .env file with API keys and DATABASE_URL
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

# Load environment variables before any other imports
from dotenv import load_dotenv

# Find project root and load .env
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    print(f"✓ Loaded environment from {ENV_FILE}")
else:
    print(f"⚠ No .env file found at {ENV_FILE}")

# Add project to path
sys.path.insert(0, str(PROJECT_ROOT))

# Now import project modules
from triplegain.src.utils.config import ConfigLoader
from triplegain.src.data.database import DatabasePool, DatabaseConfig
from triplegain.src.data.market_snapshot import MarketSnapshotBuilder
from triplegain.src.data.indicator_library import IndicatorLibrary
from triplegain.src.llm.prompt_builder import PromptBuilder
from triplegain.src.llm.clients import OllamaClient, DeepSeekClient, AnthropicClient, OpenAIClient, XAIClient
from triplegain.src.agents import TechnicalAnalysisAgent, RegimeDetectionAgent, TradingDecisionAgent
from triplegain.src.risk.rules_engine import RiskManagementEngine
from triplegain.src.orchestration.message_bus import MessageBus
from triplegain.src.orchestration.coordinator import CoordinatorAgent
from triplegain.src.execution.trading_mode import TradingMode

# Create logs directory if needed
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / 'paper_trading.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print startup banner."""
    print()
    print("=" * 60)
    print("  TripleGain Paper Trading System")
    print("  Multi-Agent LLM Trading with 6-Model Consensus")
    print("=" * 60)
    print(f"  Mode: PAPER TRADING (Simulated)")
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    print()


async def check_ollama_models() -> str:
    """Check available Ollama models and return best available."""
    import httpx

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    preferred_models = ["qwen2.5:7b", "qwen2.5:latest", "llama3.1:8b", "mistral:7b", "gemma2:9b"]

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ollama_host}/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                available = [m["name"] for m in data.get("models", [])]

                for model in preferred_models:
                    if model in available:
                        return model

                # Return first available if no preferred
                if available:
                    return available[0]

    except Exception as e:
        logger.warning(f"Could not check Ollama models: {e}")

    return "llama3.1:8b"  # Default fallback


async def init_database(config: dict) -> DatabasePool:
    """Initialize database connection pool."""
    # Get database config from environment variables
    db_config = DatabaseConfig(
        host=os.getenv("DATABASE_HOST", "localhost"),
        port=int(os.getenv("DATABASE_PORT", "5433")),
        database=os.getenv("DATABASE_NAME", "kraken_data"),
        user=os.getenv("DATABASE_USER", "trading"),
        password=os.getenv("DATABASE_PASSWORD", ""),
        min_connections=config.get("database", {}).get("connection", {}).get("min_connections", 5),
        max_connections=config.get("database", {}).get("connection", {}).get("max_connections", 20),
        command_timeout=config.get("database", {}).get("connection", {}).get("command_timeout", 60),
    )

    pool = DatabasePool(db_config)
    await pool.connect()

    # Quick health check
    async with pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1")
        if result != 1:
            raise RuntimeError("Database health check failed")

    logger.info("✓ Database pool initialized")
    return pool


def init_llm_clients() -> dict:
    """Initialize all LLM clients."""
    clients = {}

    # Ollama (local)
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    clients["ollama"] = OllamaClient({
        "base_url": ollama_host,
        "timeout_seconds": 60,
        "json_mode": True,
    })
    logger.info(f"✓ Ollama client initialized ({ollama_host})")

    # DeepSeek
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        clients["deepseek"] = DeepSeekClient({
            "api_key": deepseek_key,
            "timeout_seconds": 60,
        })
        logger.info("✓ DeepSeek client initialized")

    # Anthropic (Claude)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        clients["anthropic"] = AnthropicClient({
            "api_key": anthropic_key,
            "timeout_seconds": 60,
        })
        logger.info("✓ Anthropic client initialized")

    # OpenAI (GPT)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        clients["openai"] = OpenAIClient({
            "api_key": openai_key,
            "timeout_seconds": 60,
        })
        logger.info("✓ OpenAI client initialized")

    # xAI (Grok)
    xai_key = os.getenv("XAI_BEARER_API_KEY")
    if xai_key:
        clients["xai"] = XAIClient({
            "api_key": xai_key,
            "timeout_seconds": 60,
        })
        logger.info("✓ xAI client initialized")

    return clients


async def init_agents(
    llm_clients: dict,
    configs: dict,
    db_pool: DatabasePool,
    ollama_model: str,
) -> dict:
    """Initialize all agents."""
    agents = {}

    # Create prompt builder (prompts.yaml has nested 'prompts:' key)
    prompts_config = configs.get("prompts", {}).get("prompts", {})
    prompt_builder = PromptBuilder(prompts_config)

    # Create indicator library
    indicator_library = IndicatorLibrary(configs.get("indicators", {}))

    # Create snapshot builder
    snapshot_builder = MarketSnapshotBuilder(
        db_pool=db_pool,
        indicator_library=indicator_library,
        config=configs.get("snapshot", {}),
    )
    agents["snapshot_builder"] = snapshot_builder
    logger.info("✓ Market Snapshot Builder initialized")

    # Technical Analysis Agent (uses Ollama)
    ta_config = configs.get("agents", {}).get("technical_analysis", {})
    ta_config["model"] = ollama_model  # Override with available model
    agents["technical_analysis"] = TechnicalAnalysisAgent(
        llm_client=llm_clients["ollama"],
        prompt_builder=prompt_builder,
        config=ta_config,
        db_pool=db_pool,
    )
    logger.info(f"✓ Technical Analysis Agent initialized (model: {ollama_model})")

    # Regime Detection Agent (uses Ollama)
    regime_config = configs.get("agents", {}).get("regime_detection", {})
    regime_config["model"] = ollama_model  # Override with available model
    agents["regime_detection"] = RegimeDetectionAgent(
        llm_client=llm_clients["ollama"],
        prompt_builder=prompt_builder,
        config=regime_config,
        db_pool=db_pool,
    )
    logger.info(f"✓ Regime Detection Agent initialized (model: {ollama_model})")

    # Trading Decision Agent (uses 6-model A/B testing)
    td_config = configs.get("agents", {}).get("trading_decision", {})
    agents["trading_decision"] = TradingDecisionAgent(
        llm_clients=llm_clients,
        prompt_builder=prompt_builder,
        config=td_config,
        db_pool=db_pool,
    )
    logger.info("✓ Trading Decision Agent initialized (6-model A/B)")

    return agents


def init_risk_engine(config: dict) -> RiskManagementEngine:
    """Initialize risk management engine."""
    engine = RiskManagementEngine(config.get("risk", {}))
    logger.info("✓ Risk Management Engine initialized")
    return engine


async def main():
    """Main entry point."""
    print_banner()

    # Load configurations
    print("[1/7] Loading configuration...")
    config_loader = ConfigLoader(PROJECT_ROOT / "config")
    configs = config_loader.load_all()
    print(f"  ✓ Loaded {len(configs)} config files")

    # Initialize database
    print("[2/7] Connecting to database...")
    try:
        db_pool = await init_database(configs)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        print(f"  ✗ Database connection failed: {e}")
        print("  Make sure Docker is running: docker-compose up -d timescaledb")
        return 1

    # Check Ollama models
    print("[3/7] Checking Ollama models...")
    ollama_model = await check_ollama_models()
    print(f"  ✓ Using Ollama model: {ollama_model}")

    # Initialize LLM clients
    print("[4/7] Initializing LLM clients...")
    try:
        llm_clients = init_llm_clients()
        print(f"  ✓ Initialized {len(llm_clients)} LLM clients")
    except Exception as e:
        logger.error(f"LLM client initialization failed: {e}")
        print(f"  ✗ LLM client initialization failed: {e}")
        await db_pool.disconnect()
        return 1

    # Initialize agents
    print("[5/7] Initializing agents...")
    try:
        agents = await init_agents(llm_clients, configs, db_pool, ollama_model)
        print(f"  ✓ Initialized {len(agents)} agents")
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        print(f"  ✗ Agent initialization failed: {e}")
        await db_pool.disconnect()
        return 1

    # Initialize risk engine
    print("[6/7] Initializing risk engine...")
    risk_engine = init_risk_engine(configs)

    # Initialize coordinator
    print("[7/7] Starting coordinator...")
    message_bus = MessageBus()

    coordinator = CoordinatorAgent(
        message_bus=message_bus,
        agents=agents,
        llm_client=llm_clients.get("deepseek") or llm_clients.get("anthropic"),
        config=configs.get("orchestration", {}),
        risk_engine=risk_engine,
        execution_manager=None,  # Paper trading doesn't need real execution
        db_pool=db_pool,
        trading_mode=TradingMode.PAPER,
        execution_config=configs.get("execution", {}),
    )

    # Setup shutdown handler
    shutdown_event = asyncio.Event()

    def shutdown_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        # Start coordinator
        await coordinator.start()

        print()
        print("=" * 60)
        print("  Paper Trading System Running")
        print("  Press Ctrl+C to stop")
        print("=" * 60)
        print()

        # Show initial portfolio
        if coordinator.paper_portfolio:
            balances = coordinator.paper_portfolio.get_balances_dict()
            print("Initial Portfolio:")
            for asset, balance in balances.items():
                print(f"  {asset}: {balance:,.6f}")
            print()

        # Wait for shutdown signal
        await shutdown_event.wait()

    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
        return 1
    finally:
        print("\nShutting down...")
        await coordinator.stop()
        await db_pool.disconnect()
        print("✓ Shutdown complete")

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
