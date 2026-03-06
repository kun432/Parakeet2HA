"""
日本語インテントパーサー（HA連携なし）
シンプルなウェイクワード判定とインテント分類を行う
"""
import logging
import difflib
import re
from dataclasses import dataclass
from typing import Optional, List, Dict

logger = logging.getLogger("IntentParser_JA")


@dataclass
class Intent:
    """認識されたインテント"""
    action: str
    domain: Optional[str] = None
    entity: Optional[str] = None
    parameters: Optional[Dict] = None


class IntentParserJA:
    """日本語専用インテントパーサー"""

    def __init__(self, config: dict):
        asr_config = config.get("asr_settings", {})

        self.wake_words = asr_config.get("wake_words", ["ドライブ", "ねえ、ドライブ"])
        self.authoritative = asr_config.get("wake_word_authoritative", True)
        self.strictness = asr_config.get("wake_word_strictness", 0.60)

        # アクションマップ（日本語 -> サービス名）
        self.action_map = config.get("action_map", {
            "つけて": "turn_on",
            "消して": "turn_off",
            "切り替えて": "toggle",
            "開けて": "open",
            "閉じて": "close",
            "止めて": "stop",
        })

        # ドメインエイリアス（日本語別名 -> ドメイン）
        self.domain_aliases = config.get("domain_aliases", {
            "light": ["照明", "ライト", "電気"],
            "switch": ["スイッチ", "コンセント"],
            "cover": ["ブラインド", "カーテン", "シャッター"],
            "fan": ["扇風機", "ファン"],
        })

        # 逆引きマップ（日本語 -> ドメイン）
        self.alias_to_domain = {}
        for domain, aliases in self.domain_aliases.items():
            for alias in aliases:
                self.alias_to_domain[alias] = domain

    def check_wake_word(self, text: str) -> Optional[tuple]:
        """
        ウェイクワード検出のみ（ツーショット用）

        Args:
            text: 認識されたテキスト

        Returns:
            (matched_ww, score) or None
        """
        if not text:
            return None

        clean_text = text.strip()

        for ww in self.wake_words:
            # 完全一致
            if clean_text == ww:
                logger.info(f"✅ ウェイクワード検出: '{ww}' (完全一致)")
                return (ww, 1.0)

            # 先頭マッチング
            if clean_text.startswith(ww):
                return (ww, 1.0)

            # ファジーマッチング
            if self.authoritative:
                score = difflib.SequenceMatcher(None, ww, clean_text).ratio()
                if score >= self.strictness:
                    logger.info(f"✅ ウェイクワード検出: '{ww}' (スコア: {score:.2f})")
                    return (ww, score)

        return None

    def parse_intent(self, text: str) -> Optional[Intent]:
        """
        インテント解析のみ（ウェイクワードチェックなし、ツーショット用）

        Args:
            text: 認識されたテキスト

        Returns:
            Intent or None
        """
        if not text:
            return None

        clean_text = text.strip()
        logger.info(f"🔍 インテント解析: '{clean_text}'")

        # ==========================================
        # STEP 1: アクション抽出
        # ==========================================
        action, remaining_text = self._extract_action(clean_text)
        if not action:
            action = "toggle"  # デフォルト
            logger.info(f"⚠️ アクション未検出、デフォルト: {action}")
        else:
            logger.info(f"🎯 アクション: {action}")

        # ==========================================
        # STEP 2: 数値パラメータ抽出
        # ==========================================
        number = self._extract_number(remaining_text)
        parameters = {}
        if number is not None:
            parameters["value"] = number
            logger.info(f"🔢 パラメータ: {number}")

        # ==========================================
        # STEP 3: ドメイン/エンティティ抽出
        # ==========================================
        domain, entity = self._extract_domain_and_entity(remaining_text)
        if domain:
            logger.info(f"🏠 ドメイン: {domain}")
        if entity:
            logger.info(f"📦 エンティティ: {entity}")

        return Intent(
            action=action,
            domain=domain,
            entity=entity,
            parameters=parameters if parameters else None
        )

    def parse(self, text: str) -> Optional[Intent]:
        """
        テキストを解析してインテントを返す

        Args:
            text: 認識されたテキスト

        Returns:
            Intent or None（ウェイクワードが検出されなかった場合）
        """
        if not text:
            return None

        # 正規化
        clean_text = text.strip()
        logger.info(f"🔍 解析中: '{clean_text}'")

        # ==========================================
        # STEP 1: ウェイクワード判定
        # ==========================================
        wake_word_match = self._check_wake_word(clean_text)
        if wake_word_match is None:
            logger.warning("🚫 ウェイクワードが検出されませんでした")
            return None

        matched_ww, score, remaining_text = wake_word_match
        logger.info(f"✅ ウェイクワード検出: '{matched_ww}' (スコア: {score:.2f})")

        # ==========================================
        # STEP 2: アクション抽出
        # ==========================================
        action, remaining_text = self._extract_action(remaining_text)
        if not action:
            action = "toggle"  # デフォルト
            logger.info(f"⚠️ アクション未検出、デフォルト: {action}")
        else:
            logger.info(f"🎯 アクション: {action}")

        # ==========================================
        # STEP 3: 数値パラメータ抽出
        # ==========================================
        number = self._extract_number(remaining_text)
        parameters = {}
        if number is not None:
            parameters["value"] = number
            logger.info(f"🔢 パラメータ: {number}")

        # ==========================================
        # STEP 4: ドメイン/エンティティ抽出
        # ==========================================
        domain, entity = self._extract_domain_and_entity(remaining_text)
        if domain:
            logger.info(f"🏠 ドメイン: {domain}")
        if entity:
            logger.info(f"📦 エンティティ: {entity}")

        return Intent(
            action=action,
            domain=domain,
            entity=entity,
            parameters=parameters if parameters else None
        )

    def _check_wake_word(self, text: str) -> Optional[tuple]:
        """
        ウェイクワードをチェック

        Returns:
            (matched_ww, score, remaining_text) or None
        """
        text_lower = text

        for ww in self.wake_words:
            # 先頭マッチング
            if text_lower.startswith(ww):
                remaining = text[len(ww):].strip()
                return (ww, 1.0, remaining)

            # ファジーマッチング（authoritativeモード）
            if self.authoritative:
                # 文の先頭部分と比較
                words = text_lower.split()
                ww_words = ww.split()

                if len(words) >= len(ww_words):
                    front_chunk = " ".join(words[:len(ww_words)])
                    score = difflib.SequenceMatcher(None, ww, front_chunk).ratio()

                    if score >= self.strictness:
                        remaining = text[len(front_chunk):].strip()
                        return (ww, score, remaining)

        return None

    def _extract_action(self, text: str) -> tuple:
        """
        アクション語を抽出

        Returns:
            (action, remaining_text)
        """
        remaining = text

        # 長いフレーズから優先的にマッチ
        sorted_phrases = sorted(self.action_map.keys(), key=len, reverse=True)

        for phrase in sorted_phrases:
            if phrase in remaining:
                action = self.action_map[phrase]
                remaining = remaining.replace(phrase, "", 1).strip()
                return (action, remaining)

        return (None, remaining)

    def _extract_number(self, text: str) -> Optional[int]:
        """数値を抽出"""
        match = re.search(r'\d+', text)
        if match:
            return int(match.group())
        return None

    def _extract_domain_and_entity(self, text: str) -> tuple:
        """
        ドメインとエンティティを抽出

        Returns:
            (domain, entity)
        """
        domain = None
        entity = None

        # ドメインエイリアスを検索
        for alias, dom in sorted(self.alias_to_domain.items(), key=lambda x: len(x[0]), reverse=True):
            if alias in text:
                domain = dom
                # エイリアスを除去してエンティティ候補を探す
                entity_part = text.replace(alias, "").strip()
                if entity_part:
                    entity = entity_part
                break

        # ドメインが見つからない場合、残りのテキストをエンティティとして扱う
        if not domain and text:
            entity = text

        return (domain, entity)
