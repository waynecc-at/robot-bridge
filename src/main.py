"""Robot Bridge - Main Entry Point"""
import asyncio
import sys
import signal
from pathlib import Path
from loguru import logger

import uvicorn
from uvicorn.config import Config

from .config import config
from .hermes_client import HermesClient


def setup_logging():
    """Configure logging"""
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=config.log.level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )
    
    # Add file handler
    log_file = Path(config.log.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        level=config.log.level,
        rotation=config.log.rotation,
        retention=config.log.retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
    )
    
    logger.info(f"Logging configured, level={config.log.level}")


async def check_hermes_connection():
    """Check if Hermes Gateway is reachable"""
    logger.info(f"[Init] Checking Hermes connection at {config.hermes.base_url}...")
    
    try:
        async with HermesClient() as hermes:
            healthy = await hermes.check_health()
            if healthy:
                logger.info("[Init] Hermes Gateway is healthy")
                return True
            else:
                logger.warning("[Init] Hermes Gateway health check failed")
                return False
    except Exception as e:
        logger.error(f"[Init] Cannot connect to Hermes Gateway: {e}")
        logger.warning("[Init] Continuing anyway - Hermes may start later")
        return False


def run_server():
    """Run the Robot Bridge server"""
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("Robot Bridge - StackChan to Hermes Gateway Bridge")
    logger.info("=" * 60)
    
    # Check Hermes connection
    asyncio.run(check_hermes_connection())
    
    # Server configuration
    server_config = uvicorn.Config(
        "src.api:app",
        host=config.server.host,
        port=config.server.port,
        log_level="info",
        access_log=True,
    )
    
    server = uvicorn.Server(server_config)
    
    # Handle shutdown signals
    def shutdown_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        server.should_exit = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, shutdown_handler)
    
    logger.info(f"[Server] Starting on {config.server.host}:{config.server.port}")
    logger.info(f"[Server] WebSocket endpoint: ws://{config.server.host}:{config.server.port}{config.server.websocket_path}")
    logger.info(f"[Server] HTTP API: http://{config.server.host}:{config.server.port}/")
    logger.info("")
    logger.info("Available endpoints:")
    logger.info("  GET  /health           - Health check")
    logger.info("  POST /api/chat         - Chat with Hermes")
    logger.info("  POST /api/tts          - Text to Speech")
    logger.info("  POST /api/tts/stream   - TTS Streaming")
    logger.info("  GET  /api/voices       - List TTS voices")
    logger.info("  WS   /ws/robot         - WebSocket for ESP32")
    logger.info("")
    
    # Run server
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    finally:
        logger.info("Robot Bridge shutdown complete")


def main():
    """CLI entry point"""
    run_server()


if __name__ == "__main__":
    main()
