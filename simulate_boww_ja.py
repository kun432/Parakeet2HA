#!/usr/bin/env python3
"""
BoWWSimulator 日本語版（ツーショット対応）
マイク入力 → Silero VAD → WAV保存 → Parakeet2HA日本語版に送信

ツーショット動作:
1. 「ドライブ」→ ウェイクワード検出 → 待機状態
2. 「ライトつけて」→ インテント処理
"""
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [BoWW_JA] - %(message)s'
)
logger = logging.getLogger("BoWW_JA")


class BoWWSimulatorJA:
    """日本語版 BoWW シミュレーター"""

    def __init__(self):
        # 音声設定（Sileroは16kHz必須）
        self.RATE = 16000
        self.CHUNK = 512
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1

        # VAD設定
        self.VAD_THRESHOLD = 0.5
        self.SILENCE_LIMIT_SEC = 1.0
        self.silence_limit_chunks = int((self.RATE / self.CHUNK) * self.SILENCE_LIMIT_SEC)

        # プリロールバッファ（発話開始前の音声を保持）
        self.PRE_ROLL_SEC = 0.5
        self.pre_roll_chunks = int((self.RATE / self.CHUNK) * self.PRE_ROLL_SEC)
        self.ring_buffer = deque(maxlen=self.pre_roll_chunks)

        # 出力先
        self.spool_dir = "/tmp/boww_spool_ja"
        self.socket_path = "/tmp/parakeet2ha_ja.sock"

        # クライアント情報（ダミー）
        self.client_id = "test-client-001"
        self.group_name = "default"

        if not os.path.exists(self.spool_dir):
            os.makedirs(self.spool_dir)

        logger.info("📥 Silero VADモデル読み込み中...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.vad_model.eval()
        logger.info("✅ VADモデル読み込み完了")

        # オーディオ初期化
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

    def is_speech(self, audio_chunk: bytes) -> float:
        """Silero VADで音声判定"""
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

        with torch.no_grad():
            confidence = self.vad_model(audio_tensor, self.RATE).item()
        return confidence

    def send_to_parakeet(self, json_path: str):
        """Parakeet2HA日本語版にIPC送信"""
        logger.info(f"🔗 Parakeet2HA日本語版に送信中...")
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(self.socket_path)

            message = f"PROCESS:{json_path}\n"
            client.sendall(message.encode('utf-8'))

            response = client.recv(1024).decode('utf-8').strip()
            logger.info(f"📥 応答: {response}")

            if response == "BUSY":
                logger.info("⏳ 処理中...")
                final_response = client.recv(1024).decode('utf-8').strip()
                logger.info(f"🔓 処理完了: {final_response}")

            client.close()

        except FileNotFoundError:
            logger.error(f"❌ ソケットに接続できません: {self.socket_path}")
            logger.error("   main_ja.py が起動していますか？")
        except Exception as e:
            logger.error(f"❌ IPCエラー: {e}")

    def run(self):
        """メインループ"""
        logger.info("🎙️ マイク待機中... ウェイクワード「アレクサ」または「ねえ、アレクサ」")
        logger.info("   終了: Ctrl+C")

        recording = False
        frames = []
        silence_counter = 0

        try:
            while True:
                chunk = self.stream.read(self.CHUNK, exception_on_overflow=False)
                if not chunk:
                    continue

                confidence = self.is_speech(chunk)

                # VUメーター表示
                chunk_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                chunk_np -= np.mean(chunk_np)  # DCオフセット除去
                peak_energy = np.abs(chunk_np).max()

                bar_length = 30
                scaled_energy = min(int((peak_energy / 10000.0) * bar_length), bar_length)
                vu_bar = "█" * scaled_energy + "░" * (bar_length - scaled_energy)

                status_tag = "🔴 録音" if recording else "🟢 待機"
                sys.stdout.write(f"\r{status_tag} [{vu_bar}] 音量:{peak_energy:5.0f} | VAD:{confidence:4.2f}\033[K")
                sys.stdout.flush()

                if confidence > self.VAD_THRESHOLD:
                    if not recording:
                        sys.stdout.write("\n")
                        logger.info("🗣️ 音声検出！録音開始...")
                        recording = True

                        # プリロールバッファを追加
                        frames.extend(self.ring_buffer)
                        self.ring_buffer.clear()

                    frames.append(chunk)
                    silence_counter = 0

                elif recording:
                    frames.append(chunk)
                    silence_counter += 1

                    if silence_counter > self.silence_limit_chunks:
                        sys.stdout.write("\n")
                        logger.info("⏸️ 無音検出。処理中...")

                        timestamp = str(int(time.time()))
                        wav_path = os.path.join(self.spool_dir, f"{timestamp}.wav")
                        json_path = os.path.join(self.spool_dir, f"{timestamp}.json")

                        # WAV保存
                        with wave.open(wav_path, 'wb') as wf:
                            wf.setnchannels(self.CHANNELS)
                            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
                            wf.setframerate(self.RATE)
                            wf.writeframes(b''.join(frames))

                        # Ledger作成
                        ledger = {
                            "command_id": timestamp,
                            "audio_file": wav_path,
                            "boww_server": {
                                "status": "captured",
                                "timestamp": time.time(),
                                "client_id": self.client_id,
                                "group_name": self.group_name
                            },
                            "parakeet2ha_ja": {}
                        }

                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(ledger, f, indent=2, ensure_ascii=False)

                        logger.info(f"📁 保存完了: {wav_path}")

                        # IPC送信
                        self.send_to_parakeet(json_path)

                        # リセット
                        recording = False
                        frames = []
                        silence_counter = 0
                        logger.info("🎙️ マイク待機中...")
                else:
                    # 非録音時はリングバッファに追加
                    self.ring_buffer.append(chunk)

        except KeyboardInterrupt:
            logger.info("\n👋 終了します...")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()


if __name__ == "__main__":
    simulator = BoWWSimulatorJA()
    simulator.run()
