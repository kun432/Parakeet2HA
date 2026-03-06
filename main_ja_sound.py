#!/usr/bin/env python3
"""
Parakeet2HA 日本語版 - サウンドフィードバック付き
ツーショット対応: ウェイクワード → インテント発話
効果音: ウェイク検出/成功/タイムアウト/エラー
"""
import os
import sys
import json
import yaml
import torch
import logging
import asyncio
import contextlib
import warnings
from datetime import datetime

# NeMoのログを抑制
os.environ['NEMO_LOG_LEVEL'] = 'CRITICAL'
os.environ['WANDB_SILENT'] = 'true'
warnings.filterwarnings("ignore")

import nemo.collections.asr as nemo_asr
from nemo.utils import logging as nemo_logging

nemo_logging.setLevel(nemo_logging.CRITICAL)
logging.getLogger('lhotse').setLevel(logging.CRITICAL)
logging.getLogger('nemo_logger').setLevel(logging.CRITICAL)

from core.intent_parser_ja import IntentParserJA
from core.sound_feedback import SoundFeedback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [JA+SND] - %(message)s'
)
logger = logging.getLogger("Parakeet_JA_Sound")


def load_config(config_path: str = "config_ja.yaml") -> dict:
    """設定ファイルを読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@contextlib.contextmanager
def suppress_output():
    """NeMoのログ出力を抑制"""
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


class ParakeetJAWithSound:
    """日本語版 Parakeet ASR + インテントパーサー + サウンド（ツーショット対応）"""

    # 状態定数
    STATE_IDLE = "idle"           # 待機中（ウェイクワード待ち）
    STATE_LISTENING = "listening"  # インテント待ち

    def __init__(self, config: dict):
        self.config = config
        self.socket_path = "/tmp/parakeet2ha_ja_sound.sock"
        self.rate = 16000

        # ツーショット状態管理
        self.state = self.STATE_IDLE
        self.last_wake_time = None
        self.timeout_seconds = 10  # インテント待ちタイムアウト（秒）

        # サウンドフィードバック初期化
        sound_config = config.get("sound_settings", {})
        self.sound = SoundFeedback(
            sounds_dir=sound_config.get("sounds_dir", "sounds"),
            enabled=sound_config.get("enabled", True)
        )

        # インテントパーサー初期化
        self.parser = IntentParserJA(config)

        # Parakeetモデル読み込み
        asr_config = config.get("asr_settings", {})
        model_name = asr_config.get("model_name", "nvidia/parakeet-tdt_ctc-0.6b-ja")
        logger.info(f"📦 Parakeetモデル読み込み: {model_name}")

        try:
            with suppress_output():
                self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
                self.asr_model.eval()
                if torch.cuda.is_available():
                    self.asr_model = self.asr_model.to('cuda')
                    logger.info("🚀 CUDA使用中")
                else:
                    logger.info("💻 CPU使用中")
            logger.info("✅ モデル読み込み完了")
        except Exception as e:
            logger.error(f"❌ モデル読み込みエラー: {e}", exc_info=True)
            sys.exit(1)

    def transcribe(self, audio_path: str) -> str:
        """
        音声ファイルをテキストに変換

        Args:
            audio_path: WAVファイルパス

        Returns:
            認識されたテキスト
        """
        if not os.path.exists(audio_path):
            logger.error(f"音声ファイルが見つかりません: {audio_path}")
            return ""

        try:
            with suppress_output():
                result = self.asr_model.transcribe([audio_path])

            # 結果の抽出
            if isinstance(result, list) and len(result) > 0:
                first = result[0]
                if hasattr(first, 'text'):
                    return first.text
                return str(first)
            return str(result)
        except Exception as e:
            logger.error(f"認識エラー: {e}")
            return ""

    def process_audio(self, audio_path: str) -> dict:
        """
        音声を処理してインテントを返す（ツーショット対応＋サウンド）

        Args:
            audio_path: WAVファイルパス

        Returns:
            結果dict
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "audio_file": audio_path,
            "transcription": "",
            "state": self.state,
            "wake_word_detected": False,
            "intent": None
        }

        # 音声認識
        text = self.transcribe(audio_path)
        result["transcription"] = text

        if not text:
            logger.warning("📝 認識結果なし")
            return result

        logger.info(f"📝 認識結果: '{text}'")

        # ==========================================
        # ツーショット処理
        # ==========================================

        # タイムアウトチェック
        if self.state == self.STATE_LISTENING:
            elapsed = (datetime.now() - self.last_wake_time).total_seconds()
            if elapsed > self.timeout_seconds:
                logger.warning(f"⏰ タイムアウト（{self.timeout_seconds}秒）→ 待機状態へ")
                self.sound.play_timeout()  # 🔊 タイムアウト音
                self.state = self.STATE_IDLE
                self.last_wake_time = None

        if self.state == self.STATE_IDLE:
            # ==========================================
            # 待機状態：ウェイクワードをチェック
            # ==========================================
            wake_result = self.parser.check_wake_word(text)

            if wake_result:
                matched_ww, score = wake_result
                logger.info(f"✅ ウェイクワード検出: '{matched_ww}' (スコア: {score:.2f})")
                self.sound.play_wake()  # 🔊 ウェイクワード検出音
                print("\n🔔 ** インテント待ち **\n")

                self.state = self.STATE_LISTENING
                self.last_wake_time = datetime.now()

                result["wake_word_detected"] = True
                result["state"] = self.state
            else:
                logger.info("🚫 ウェイクワード未検出")

        elif self.state == self.STATE_LISTENING:
            # ==========================================
            # インテント待ち状態：インテントを処理
            # ==========================================
            intent = self.parser.parse_intent(text)

            if intent:
                # ドメインがない場合は不完全なインテント → エラー
                if not intent.domain:
                    logger.warning("🚫 不完全なインテント（ドメインなし）")
                    self.sound.play_error()  # 🔊 エラー音
                    # タイムアウトまではLISTENING状態を維持
                    return result

                result["wake_word_detected"] = True  # 文脈的にウェイク済み
                result["intent"] = {
                    "action": intent.action,
                    "domain": intent.domain,
                    "entity": intent.entity,
                    "parameters": intent.parameters
                }
                logger.info(f"🎯 インテント: {result['intent']}")
                self.sound.play_success()  # 🔊 成功音

                # 待機状態に戻る
                self.state = self.STATE_IDLE
                self.last_wake_time = None
                result["state"] = self.STATE_IDLE
            else:
                logger.warning("🚫 インテント未検出")
                self.sound.play_error()  # 🔊 エラー音
                # タイムアウトまではLISTENING状態を維持

        return result

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """IPCクライアント処理"""
        try:
            data = await reader.read(1024)
            message = data.decode('utf-8').strip()

            if not message.startswith("PROCESS:"):
                writer.write(b"ERROR: Invalid protocol\n")
                await writer.drain()
                return

            json_path = message.split("PROCESS:")[1].strip()

            # 処理開始
            writer.write(b"BUSY\n")
            await writer.drain()

            if not os.path.exists(json_path):
                logger.error(f"Ledgerファイルが見つかりません: {json_path}")
                writer.write(b"READY\n")
                await writer.drain()
                return

            # Ledger読み込み
            with open(json_path, 'r', encoding='utf-8') as f:
                ledger = json.load(f)

            audio_file = ledger.get("audio_file", "")
            logger.info(f"📥 処理開始: {audio_file}")

            # 音声処理
            result = self.process_audio(audio_file)

            # Ledger更新
            ledger["parakeet2ha_ja_sound"] = result

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(ledger, f, indent=2, ensure_ascii=False)

            # 結果を標準出力にも出力
            print("\n" + "=" * 50)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("=" * 50 + "\n")

            # 処理完了
            writer.write(b"READY\n")
            await writer.drain()

        except Exception as e:
            logger.error(f"IPCエラー: {e}", exc_info=True)
        finally:
            writer.close()
            await writer.wait_closed()

    async def timeout_watcher(self):
        """バックグラウンドでタイムアウトを監視"""
        while True:
            await asyncio.sleep(1)  # 1秒ごとにチェック

            if self.state == self.STATE_LISTENING and self.last_wake_time:
                elapsed = (datetime.now() - self.last_wake_time).total_seconds()
                if elapsed > self.timeout_seconds:
                    logger.warning(f"⏰ タイムアウト（{self.timeout_seconds}秒）→ 待機状態へ")
                    self.sound.play_timeout()  # 🔊 タイムアウト音
                    print("\n⏰ ** タイムアウト **\n")
                    self.state = self.STATE_IDLE
                    self.last_wake_time = None

    async def start_server(self):
        """IPCサーバー起動"""
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        server = await asyncio.start_unix_server(self.handle_client, path=self.socket_path)
        logger.info(f"🟢 IPCサーバー起動: {self.socket_path}")
        logger.info(f"🔌 ウェイクワード: {self.config.get('asr_settings', {}).get('wake_words', [])}")
        logger.info(f"🔊 サウンドフィードバック: {'有効' if self.sound.enabled else '無効'}")
        logger.info(f"⏱️ インテント待ちタイムアウト: {self.timeout_seconds}秒")

        # タイムアウト監視タスクを開始
        asyncio.create_task(self.timeout_watcher())

        async with server:
            await server.serve_forever()


def transcribe_file(model: ParakeetJAWithSound, audio_path: str, output_json: bool = True):
    """単発ファイル認識（IPCなし）"""
    result = model.process_audio(audio_path)

    if output_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"\n📝 認識結果: {result['transcription']}")
        if result['intent']:
            print(f"🎯 インテント: {result['intent']}")
        else:
            print("🚫 インテント未検出")


async def main():
    """メインエントリポイント"""
    import argparse

    parser = argparse.ArgumentParser(description="Parakeet2HA 日本語版（サウンド付き）")
    parser.add_argument("--config", "-c", default="config_ja.yaml", help="設定ファイルパス")
    parser.add_argument("--file", "-f", help="単発WAVファイル認識モード")
    parser.add_argument("--json", "-j", action="store_true", help="JSON出力")
    args = parser.parse_args()

    # 設定読み込み
    config = load_config(args.config)

    # モデル初期化
    model = ParakeetJAWithSound(config)

    # 単発ファイルモード
    if args.file:
        transcribe_file(model, args.file, output_json=args.json)
        return

    # IPCサーバーモード
    await model.start_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 終了します...")
        socket_path = "/tmp/parakeet2ha_ja_sound.sock"
        if os.path.exists(socket_path):
            os.remove(socket_path)
