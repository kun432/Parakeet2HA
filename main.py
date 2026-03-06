import asyncio
import logging

# We import the new master execution loop directly from the engine
from core.asr_engine import main_loop

if __name__ == "__main__":
    try:
        # Fire up the IPC server and Home Assistant pipeline
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
