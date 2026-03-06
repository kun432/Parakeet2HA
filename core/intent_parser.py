import logging
import difflib
import re
from dataclasses import dataclass
from core.config_loader import config

logger = logging.getLogger("IntentParser")

@dataclass
class Intent:
    domain: str
    service: str
    entity_id: str = None
    area_id: str = None
    payload: dict = None

class IntentParser:
    """A language-agnostic intent parser supporting Context-Aware Area Sweeps and Parameter Injection."""
    
    def __init__(self):
        self.wake_words = config.wake_words
        self.authoritative = config.ww_authoritative
        self.strictness = config.ww_strictness
        self.action_map = {}
        
        self.domain_aliases = config.domain_aliases or {
            "light": ["lights", "light"],
            "switch": ["switches", "switch", "plugs", "plug"],
            "cover": ["blinds", "shades", "covers", "door", "doors"],
            "fan": ["fans", "fan"]
        }

    def load_translations(self, raw_translations: dict):
        if not raw_translations:
            return

        flat_map = {}
        for key, value in raw_translations.items():
            if 'services' in key and key.endswith('.name'):
                parts = key.split('.')
                try:
                    service_idx = parts.index('services')
                    service_name = parts[service_idx + 1]
                    localized_phrase = str(value).lower()
                    flat_map[localized_phrase] = service_name
                except (ValueError, IndexError):
                    continue
        
        if flat_map:
            self.action_map = flat_map

    def _apply_parameters(self, intent: Intent, extracted_number: int) -> Intent:
        """Intelligently overrides services and builds payloads if a number was spoken."""
        if extracted_number is None:
            return intent
            
        if intent.domain == "fan":
            intent.service = "set_percentage"
            intent.payload = {"percentage": extracted_number}
            logger.info(f"⚙️ Parameter Injected: Fan -> {extracted_number}%")
            
        elif intent.domain == "light":
            intent.service = "turn_on"  
            intent.payload = {"brightness_pct": extracted_number}
            logger.info(f"⚙️ Parameter Injected: Light -> {extracted_number}%")
            
        elif intent.domain == "cover":
            intent.service = "set_cover_position"
            intent.payload = {"position": extracted_number}
            logger.info(f"⚙️ Parameter Injected: Cover -> {extracted_number}%")
            
        elif intent.domain == "climate":
            intent.service = "set_temperature"
            intent.payload = {"temperature": extracted_number}
            logger.info(f"⚙️ Parameter Injected: Climate -> {extracted_number}°")
            
        # --- NEW: Catching input_number and number helpers ---
        elif intent.domain in ["input_number", "number"]:
            intent.service = "set_value"
            # HA often expects floats for number entities
            intent.payload = {"value": float(extracted_number)} 
            logger.info(f"⚙️ Parameter Injected: Number/Input -> {extracted_number}")
            
        return intent

    def parse(self, text: str, roster: list, mapped_area: str = None) -> Intent:
        clean_text = text.lower()
        
        clean_text = re.sub(r'[^\w\s\d]', '', clean_text)
        
        logger.info(f"🧠 Analyzing: '{clean_text}' (Injected Context: {mapped_area})")

        # ==========================================
        # PARAMETER EXTRACTION
        # ==========================================
        number_match = re.search(r'\b(\d+)\b', clean_text)
        extracted_number = int(number_match.group(1)) if number_match else None

        # ==========================================
        # TIER 0: AUTHORITATIVE WAKE WORD GATEKEEPER
        # ==========================================
        if self.authoritative:
            words = clean_text.split()
            ww_found = False
            
            for ww in self.wake_words:
                ww_len = len(ww.split())
                if len(words) >= ww_len:
                    front_chunk = " ".join(words[:ww_len])
                    score = difflib.SequenceMatcher(None, ww.lower(), front_chunk).ratio()
                    
                    if score >= self.strictness:
                        ww_found = True
                        logger.info(f"🛡️ Authoritative Wake Word Verified: '{front_chunk}' (Score: {score:.2f})")
                        clean_text = clean_text.replace(front_chunk, "", 1).strip()
                        break
                        
            if not ww_found:
                logger.warning(f"🚫 Rejected: Command did not begin with a valid wake word above strictness threshold ({self.strictness}).")
                return None
        else:
            for w in self.wake_words:
                clean_text = clean_text.replace(w.lower(), "").strip()

        # 1. Extract Action (Verb)
        target_service = None
        remaining_text = clean_text
        sorted_phrases = sorted(self.action_map.keys(), key=len, reverse=True)
        
        for phrase in sorted_phrases:
            if re.search(rf'\b{phrase}\b', remaining_text):
                target_service = self.action_map[phrase]
                remaining_text = remaining_text.replace(phrase, " ").strip()
                break

        if not target_service:
            target_service = "toggle"

        # 2. Build Roster Maps
        device_map = {}
        area_map = {}
        for device in roster:
            name = device.get('name', '').lower()
            if name: device_map[name] = device
                
            ent_id = device.get('entity_id', '')
            if ent_id: 
                clean_id = ent_id.split('.')[-1].replace('_', ' ')
                device_map[clean_id] = device
                
            a_name = device.get('area_name', '').lower()
            a_id = device.get('area_id', '')
            if a_name and a_id:
                area_map[a_name] = a_id

        # 3. Resolve the True Area ID
        implicit_area_id = None
        if mapped_area:
            mapped_lower = mapped_area.lower()
            if mapped_lower in area_map.values():
                implicit_area_id = mapped_lower
            elif mapped_lower in area_map.keys():
                implicit_area_id = area_map[mapped_lower]
            else:
                for a_name, a_id in area_map.items():
                    if mapped_lower in a_name or a_name in mapped_lower:
                        implicit_area_id = a_id
                        break

        # 4. Check for Spoken Area Name
        matched_area_name = None
        for a_name in sorted(area_map.keys(), key=len, reverse=True):
            if a_name in clean_text:
                matched_area_name = a_name
                break
                
        active_area_id = area_map.get(matched_area_name) if matched_area_name else implicit_area_id

        # ==========================================
        # TIER 1: EXACT / FUZZY ENTITY SEARCH
        # ==========================================
        words = remaining_text.split()
        best_match_name = None
        best_match_score = 0.0
        possible_names = list(device_map.keys())

        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                chunk = " ".join(words[i:j])
                matches = difflib.get_close_matches(chunk, possible_names, n=1, cutoff=0.75)
                if matches:
                    raw_score = difflib.SequenceMatcher(None, chunk, matches[0]).ratio()
                    
                    matched_device = device_map[matches[0]]
                    if active_area_id and matched_device.get('area_id') == active_area_id:
                        # Cap the score at 1.0 so the logs don't look mathematically broken
                        score = min(raw_score + 0.15, 1.0)
                    else:
                        score = raw_score
                        
                    if score > best_match_score:
                        best_match_score = score
                        best_match_name = matches[0]

        if best_match_name and best_match_score > 0.8:
            target_device = device_map[best_match_name]
            logger.info(f"✨ TIER 1 MATCH: Entity '{best_match_name}' (Score: {best_match_score:.2f})")
            
            intent = Intent(
                domain=target_device.get('domain', target_device.get('entity_id', '').split('.')[0]),
                service=target_service,
                entity_id=target_device.get('entity_id'),
                area_id=None,
                payload={}
            )
            return self._apply_parameters(intent, extracted_number)

        # ==========================================
        # TIER 2: AREA SWEEP (Plurals) FALLBACK
        # ==========================================
        matched_domain = None
        for domain, aliases in self.domain_aliases.items():
            for alias in sorted(aliases, key=len, reverse=True):
                if re.search(rf'\b{alias.lower()}\b', clean_text):
                    matched_domain = domain
                    break
            if matched_domain:
                break
                
        if matched_domain and active_area_id:
            logger.info(f"✨ TIER 2 FALLBACK: Area Sweep '{active_area_id}' + Domain '{matched_domain}'")
            intent = Intent(
                domain=matched_domain,
                service=target_service,
                entity_id=None,
                area_id=active_area_id,
                payload={}
            )
            return self._apply_parameters(intent, extracted_number)

        logger.warning(f"🤔 Could not match a specific entity or a room sweep.")
        return None
