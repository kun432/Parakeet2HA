"""
サウンドフィードバックモジュール
ウェイクワード検出、インテント成功、タイムアウト、エラー時に音を鳴らす
"""
import os
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("SoundFeedback")


class SoundFeedback:
    """サウンドフィードバック管理"""

    # サウンド種別
    WAKE = "wake"        # ウェイクワード検出
    SUCCESS = "success"  # インテント成功
    TIMEOUT = "timeout"  # タイムアウト
    ERROR = "error"      # インテント不適切

    def __init__(self, sounds_dir: str = "sounds", enabled: bool = True):
        """
        Args:
            sounds_dir: サウンドファイル格納ディレクトリ
            enabled: サウンド有効/無効
        """
        self.sounds_dir = Path(sounds_dir)
        self.enabled = enabled

        # サウンドファイルパス
        self.sound_files = {
            self.WAKE: self.sounds_dir / "wake.wav",
            self.SUCCESS: self.sounds_dir / "success.wav",
            self.TIMEOUT: self.sounds_dir / "timeout.wav",
            self.ERROR: self.sounds_dir / "error.wav",
        }

        if self.enabled:
            self._check_sound_files()

    def _check_sound_files(self):
        """サウンドファイルの存在確認"""
        for sound_type, path in self.sound_files.items():
            if path.exists():
                logger.info(f"🔊 {sound_type}: {path}")
            else:
                logger.warning(f"⚠️ {sound_type}: ファイルなし ({path})")

    def play(self, sound_type: str) -> bool:
        """
        サウンドを再生

        Args:
            sound_type: サウンド種別 (WAKE, SUCCESS, TIMEOUT, ERROR)

        Returns:
            再生成功時True
        """
        if not self.enabled:
            return False

        path = self.sound_files.get(sound_type)
        if not path or not path.exists():
            logger.warning(f"🔇 サウンドファイルなし: {sound_type}")
            return False

        try:
            # macOS: afplayコマンドで再生（非同期）
            subprocess.Popen(
                ["afplay", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info(f"🔊 再生: {sound_type}")
            return True
        except Exception as e:
            logger.error(f"❌ 再生エラー: {e}")
            return False

    def play_wake(self) -> bool:
        """ウェイクワード検出音"""
        return self.play(self.WAKE)

    def play_success(self) -> bool:
        """インテント成功音"""
        return self.play(self.SUCCESS)

    def play_timeout(self) -> bool:
        """タイムアウト音"""
        return self.play(self.TIMEOUT)

    def play_error(self) -> bool:
        """エラー音"""
        return self.play(self.ERROR)
