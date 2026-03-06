import os
import sys
import time
import json
import socket
import wave
import pyaudio
import numpy as np
import torch
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - BoWW_SIMULATOR - %(message)s')
logger = logging.getLogger("BoWW")

class BoWWSimulator:
    def __init__(self):
        # Audio Configuration (Silero requires 16kHz)
        self.RATE = 16000
        self.CHUNK = 512
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        
        # VAD Configuration
        self.VAD_THRESHOLD = 0.5  
        self.SILENCE_LIMIT_SEC = 1.0  
        self.silence_limit_chunks = int((self.RATE / self.CHUNK) * self.SILENCE_LIMIT_SEC)
        
        # --- NEW: Pre-Roll Ring Buffer Configuration ---
        self.PRE_ROLL_SEC = 0.5 # Keep 500ms of audio before speech starts
        self.pre_roll_chunks = int((self.RATE / self.CHUNK) * self.PRE_ROLL_SEC)
        self.ring_buffer = deque(maxlen=self.pre_roll_chunks)
        
        # Output Configuration
        self.spool_dir = "/tmp/boww_spool"
        self.socket_path = "/tmp/parakeet2ha.sock"
        
        # Fake Hardware Data
        self.client_id = "guid-esp32-kitchen-001"
        self.group_name = "kitchen"
        
        if not os.path.exists(self.spool_dir):
            os.makedirs(self.spool_dir)

        logger.info("Downloading/Loading Silero VAD model...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.vad_model.eval()

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

    def is_speech(self, audio_chunk: bytes) -> float:
        """Converts the raw bytes to a tensor and gets the Silero VAD confidence score."""
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        
        with torch.no_grad():
            confidence = self.vad_model(audio_tensor, self.RATE).item()
        return confidence

    def send_to_parakeet(self, json_path: str):
        """Connects to the Parakeet2HA Unix Socket and sends the IPC ping."""
        logger.info(f"🔗 Pinging Parakeet2HA IPC Socket...")
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(self.socket_path)
            
            message = f"PROCESS:{json_path}\n"
            client.sendall(message.encode('utf-8'))
            
            response = client.recv(1024).decode('utf-8').strip()
            logger.info(f"📥 Parakeet2HA replied: {response}")
            
            if response == "BUSY":
                logger.info("⏳ Waiting for Parakeet2HA to finish processing...")
                final_response = client.recv(1024).decode('utf-8').strip()
                logger.info(f"🔓 Parakeet2HA released lock: {final_response}")
                
            client.close()
            
        except FileNotFoundError:
            logger.error(f"❌ Could not connect to {self.socket_path}. Is Parakeet2HA running?")
        except Exception as e:
            logger.error(f"IPC Error: {e}")

    def run(self):
        logger.info("🎙️ Silero VAD Listening. Say 'Hey Jarvis...'")
        
        recording = False
        frames = []
        silence_counter = 0

        try:
            while True:
                chunk = self.stream.read(self.CHUNK, exception_on_overflow=False)
                if not chunk:
                    continue
                    
                confidence = self.is_speech(chunk)

                # --- LIVE TERMINAL VU & VAD METER ---
                chunk_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                chunk_np -= np.mean(chunk_np) # DC Offset removal
                peak_energy = np.abs(chunk_np).max()
                
                bar_length = 30
                scaled_energy = min(int((peak_energy / 10000.0) * bar_length), bar_length)
                vu_bar = "█" * scaled_energy + "░" * (bar_length - scaled_energy)
                
                status_tag = "🔴 REC" if recording else "🟢 LIS"
                
                sys.stdout.write(f"\r{status_tag} [{vu_bar}] VOL:{peak_energy:5.0f} | VAD:{confidence:4.2f}\033[K")
                sys.stdout.flush()
                # ------------------------------------

                if confidence > self.VAD_THRESHOLD:
                    if not recording:
                        sys.stdout.write("\n") 
                        logger.info("🗣️ Speech Detected! Dumping Pre-Roll Buffer and Recording...")
                        recording = True
                        
                        # MAGIC FIX: Dump the pre-roll ring buffer into the final audio frames
                        frames.extend(self.ring_buffer)
                        self.ring_buffer.clear()
                        
                    frames.append(chunk)
                    silence_counter = 0  
                
                elif recording:
                    frames.append(chunk)
                    silence_counter += 1
                    
                    if silence_counter > self.silence_limit_chunks:
                        sys.stdout.write("\n")
                        logger.info("⏸️ Silence Detected. Packaging command...")
                        
                        timestamp = str(int(time.time()))
                        wav_path = os.path.join(self.spool_dir, f"{timestamp}.wav")
                        json_path = os.path.join(self.spool_dir, f"{timestamp}.json")
                        
                        # Save the WAV
                        with wave.open(wav_path, 'wb') as wf:
                            wf.setnchannels(self.CHANNELS)
                            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                            wf.setframerate(self.RATE)
                            wf.writeframes(b''.join(frames))
                            
                        # Generate Ledger
                        ledger = {
                            "command_id": timestamp,
                            "audio_file": wav_path,
                            "boww_server": {
                                "status": "captured",
                                "timestamp": time.time(),
                                "client_id": self.client_id,
                                "group_name": self.group_name
                            },
                            "parakeet2ha": {}
                        }
                        
                        with open(json_path, 'w') as f:
                            json.dump(ledger, f, indent=2)
                            
                        logger.info(f"📁 Saved to {wav_path}")
                        
                        # Fire IPC
                        self.send_to_parakeet(json_path)
                        
                        recording = False
                        frames = []
                        silence_counter = 0
                        logger.info("🎙️ Silero VAD Listening...")
                else:
                    # If we are NOT recording and it's NOT speech, keep rolling the buffer
                    self.ring_buffer.append(chunk)

        except KeyboardInterrupt:
            logger.info("\nShutting down simulator...")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

if __name__ == "__main__":
    simulator = BoWWSimulator()
    simulator.run()
