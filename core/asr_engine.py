import os
import sys
import json
import numpy as np
import torch
import logging
import asyncio
import soundfile as sf
import contextlib
import warnings

# Gag NeMo's environment-level logging
os.environ['NEMO_LOG_LEVEL'] = 'CRITICAL'
os.environ['WANDB_SILENT'] = 'true'

warnings.filterwarnings("ignore")

import nemo.collections.asr as nemo_asr
from nemo.utils import logging as nemo_logging

nemo_logging.setLevel(nemo_logging.CRITICAL)
logging.getLogger('lhotse').setLevel(logging.CRITICAL)
logging.getLogger('nemo_logger').setLevel(logging.CRITICAL)

# Import our custom modules
from core.config_loader import config
from core.ha_client import HomeAssistantClient
from core.intent_parser import IntentParser
from core.lm_builder import LanguageModelBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - IPC_WORKER - %(message)s')
logger = logging.getLogger("Parakeet_IPC")

@contextlib.contextmanager
def suppress_output():
    """Aggressively silences both Python and C-level console output."""
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull)

class ParakeetIPCServer:
    """Listens on a Unix socket for JSON files from BoWWServer and processes them."""
    
    def __init__(self, ha_client: HomeAssistantClient, intent_parser: IntentParser):
        self.ha = ha_client
        self.parser = intent_parser
        self.socket_path = "/tmp/parakeet2ha.sock"
        self.rate = 16000
        
        logger.info(f"Loading Parakeet Model: {config.asr_model} ...")
        try:
            with suppress_output():
                self.asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=config.asr_model)
                self.asr_model.eval()
                if torch.cuda.is_available():
                    self.asr_model = self.asr_model.to('cuda')
            logger.info("Acoustic Model loaded.")
        except Exception as e:
            logger.error(f"CRITICAL ERROR loading Parakeet model: {e}", exc_info=True)
            sys.exit(1)

    def inject_language_model(self, lm_path: str):
        """Switches Parakeet to use the Flashlight decoder with our custom KenLM."""
        if not os.path.exists(lm_path):
            logger.warning("KenLM binary not found. Falling back to Greedy Decoding.")
            return

        logger.info(f"Fusing Acoustic Model with KenLM N-gram: {lm_path}")
        
        try:
            decoding_config = nemo_asr.modules.RNNTDecodingConfig(
                strategy="flashlight",
                beam=nemo_asr.modules.RNNTBeamConfig(
                    beam_size=32,
                    flashlight_cfg=nemo_asr.modules.FlashlightConfig(
                        lm_path=lm_path,
                        lm_weight=2.0,
                        word_score=-1.0,
                    )
                )
            )
            self.asr_model.change_decoding_strategy(decoding_config)
            logger.info("🟢 Flashlight Decoder successfully bound.")
        except Exception as e:
            logger.error(f"Failed to bind Flashlight decoder: {e}")

    def _transcribe_file(self, audio_path: str) -> str:
        """Transcribes a direct WAV file."""
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return ""
            
        try:
            with suppress_output():
                transcriptions = self.asr_model.transcribe(audio=[audio_path], verbose=False)
            
            if isinstance(transcriptions, tuple):
                transcriptions = transcriptions[0]
            while isinstance(transcriptions, list) and len(transcriptions) > 0:
                transcriptions = transcriptions[0]
            if hasattr(transcriptions, 'text'):
                return str(transcriptions.text)
            return str(transcriptions)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """The core execution pipeline triggered by BoWWServer IPC pings."""
        try:
            data = await reader.read(1024)
            message = data.decode('utf-8').strip()
            
            if not message.startswith("PROCESS:"):
                writer.write(b"ERROR: Invalid protocol\n")
                await writer.drain()
                return
                
            json_path = message.split("PROCESS:")[1].strip()
            
            # 1. Lock the pipeline
            writer.write(b"BUSY\n")
            await writer.drain()
            
            if not os.path.exists(json_path):
                logger.error(f"Ledger file missing: {json_path}")
                writer.write(b"READY\n")
                await writer.drain()
                return

            # 2. Read the BoWWServer Ledger
            with open(json_path, 'r') as f:
                ledger = json.load(f)
                
            audio_file = ledger.get("audio_file", "")
            group_name = ledger.get("boww_server", {}).get("group_name", "")
            
            logger.info(f"📥 Processing {audio_file} (Group: {group_name})")

            # 3. Transcribe the WAV
            text = await asyncio.to_thread(self._transcribe_file, audio_file)
            logger.info(f"✅ CAPTURED: '{text}'")

            # 4. Parse Intent
            # Map the BoWWServer group name into the parser context
            mapped_area = group_name 
            
            # ---> THE FIX: Passed mapped_area as the third argument! <---
            intent = self.parser.parse(text, self.ha.voice_roster, mapped_area)
            
            # 5. Execute and update ledger
            execution_status = "failed"
            if intent:
                logger.info(f"🚀 Sending -> Domain: '{intent.domain}', Service: '{intent.service}', Entity: '{intent.entity_id}' | Area: '{intent.area_id}'")
                success = await self.ha.execute_service(
                    domain=intent.domain, 
                    service=intent.service, 
                    entity_id=intent.entity_id, 
                    area_id=intent.area_id,
                    payload=intent.payload
                )
                if success:
                    execution_status = "success"
                    logger.info("🟢 SUCCESS: Request completed.")

            ledger["parakeet2ha"] = {
                "status": execution_status,
                "mapped_ha_area": mapped_area,
                "transcription": text,
                "intent": {
                    "domain": intent.domain if intent else "",
                    "service": intent.service if intent else "",
                    "entity_id": intent.entity_id if intent else "",
                    "area_id": intent.area_id if intent else ""
                }
            }
            
            with open(json_path, 'w') as f:
                json.dump(ledger, f, indent=2)

            # 6. Release the Lock
            writer.write(b"READY\n")
            await writer.drain()

        except Exception as e:
            logger.error(f"IPC connection error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def start_server(self):
        """Binds the Unix socket and listens for BoWWServer forever."""
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
            
        server = await asyncio.start_unix_server(self.handle_client, path=self.socket_path)
        logger.info(f"🟢 IPC SERVER ONLINE. Listening on {self.socket_path}")
        
        async with server:
            await server.serve_forever()

# =============================================================================
# MAIN RUNNER 
# =============================================================================
async def main_loop():
    ha = HomeAssistantClient()
    intent_parser = IntentParser()
    
    logger.info("Initiating Home Assistant connection...")
    if await ha.bootstrap_system():
        
        # --- INITIALIZATION PHASE ---
        logger.info("Compiling translations and language models...")
        
        # Pull translations from HA and load them into the parser ONCE
        intent_parser.load_translations(ha.translations)
        
        # Build the N-gram Language Model dynamically based on HA native services
        lm_builder = LanguageModelBuilder()
        lm_builder.build_corpus(ha.voice_roster, ha.domain_services, ha.translations)
        
        bin_path = "models/ha_model.bin" 
        
        # --- EXECUTION PHASE ---
        ipc_worker = ParakeetIPCServer(ha, intent_parser)
        
        # Mount the custom Language model for maximum accuracy
        if os.path.exists(bin_path):
            ipc_worker.inject_language_model(bin_path)
            
        await ipc_worker.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        if os.path.exists("/tmp/parakeet2ha.sock"):
            os.remove("/tmp/parakeet2ha.sock")
