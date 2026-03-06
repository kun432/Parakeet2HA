import os
import logging
import re
from core.config_loader import config

logger = logging.getLogger("LM_Builder")

class LanguageModelBuilder:
    """Generates a dynamic, multi-lingual smart home text corpus for KenLM."""
    
    def __init__(self, output_dir: str = "models"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.corpus_path = os.path.join(self.output_dir, "corpus.txt")
        self.user_guide_path = os.path.join(self.output_dir, "expected_commands.txt")
        self.domain_aliases = config.domain_aliases
        self.wake_words = config.wake_words
        self.authoritative = config.ww_authoritative
        
        # Templates for specific entities
        self.area_templates = [
            "{action} {area} {device}",
            "{area} {device} {action}",
            "{action} {device} {area}"
        ]
        
        self.base_templates = [
            "{action} {device}",
            "{device} {action}"
        ]
        
        # Templates for Area-Wide Sweeps (Plurals)
        self.sweep_templates = [
            "{action} the {area} {alias}",
            "{action} {area} {alias}",
            "{action} the {alias} in the {area}",
            "{area} {alias} {action}"
        ]

    def _get_localized_actions(self, domain: str, domain_services: dict, translations: dict) -> list:
        valid_services = domain_services.get(domain, [])
        spoken_actions = set()
        
        for service in valid_services:
            trans_key = f"component.{domain}.services.{service}.name"
            localized_phrase = translations.get(trans_key)
            if localized_phrase:
                spoken_actions.add(str(localized_phrase).lower())
                
        if not spoken_actions:
            for service in valid_services:
                spoken_actions.add(service.replace("_", " ").lower())
                
        return list(spoken_actions)

    def _apply_wake_words(self, sentence: str) -> list:
        """Prepends wake words based on the authoritative setting."""
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        permutations = []
        
        if self.authoritative:
            # ONLY generate sentences starting with a wake word
            for ww in self.wake_words:
                permutations.append(f"{ww} {sentence}")
        else:
            # Generate raw sentences AND wake word variations just in case
            permutations.append(sentence)
            for ww in self.wake_words:
                permutations.append(f"{ww} {sentence}")
                
        return permutations

    def build_corpus(self, roster: list, domain_services: dict, translations: dict) -> str:
        logger.info(f"Generating language permutations (Authoritative Wake Word: {self.authoritative})...")
        
        sentences = set()
        used_actions = set()
        used_devices = set()
        used_areas = set()
        
        # 1. Generate Entity-Specific Commands
        for entity in roster:
            domain = entity.get('domain', '')
            if not domain: continue
            
            valid_actions = self._get_localized_actions(domain, domain_services, translations)
            used_actions.update(valid_actions)
            
            device_names = []
            name = entity.get('name', '')
            if name: device_names.append(name.lower())
                
            clean_id = entity.get('entity_id', '').split('.')[-1].replace('_', ' ')
            device_names.append(clean_id)
            
            area = entity.get('area_name', '').lower()
            if area: used_areas.add(area)
            used_devices.update(device_names)

            for action in valid_actions:
                for device in device_names:
                    if area:
                        for template in self.area_templates:
                            raw_sentence = template.format(action=action, area=area, device=device)
                            sentences.update(self._apply_wake_words(raw_sentence))
                            
                    for template in self.base_templates:
                        raw_sentence = template.format(action=action, device=device)
                        sentences.update(self._apply_wake_words(raw_sentence))
                        
        # 2. Generate Area-Wide Sweep Commands (using YAML plurals)
        for domain, aliases in self.domain_aliases.items():
            valid_actions = self._get_localized_actions(domain, domain_services, translations)
            for alias in aliases:
                for action in valid_actions:
                    for template in self.base_templates:
                        raw_sentence = template.format(action=action, device=alias)
                        sentences.update(self._apply_wake_words(raw_sentence))
                    
                    for area in used_areas:
                        if not area: continue
                        for template in self.sweep_templates:
                            raw_sentence = template.format(action=action, area=area, alias=alias)
                            sentences.update(self._apply_wake_words(raw_sentence))

        sorted_sentences = sorted(list(sentences))

        with open(self.corpus_path, "w", encoding="utf-8") as f:
            for s in sorted_sentences:
                f.write(s + "\n")
                
        self._write_user_guide(sorted_sentences, list(used_actions), list(used_devices), used_areas)
        logger.info(f"Successfully generated {len(sentences)} permutations.")
        return self.corpus_path

    def _write_user_guide(self, sentences: list, actions: list, devices: list, areas: set):
        with open(self.user_guide_path, "w", encoding="utf-8") as f:
            f.write("--- PARAKEET2HA EXPECTED COMMANDS ---\n")
            for s in sentences:
                f.write(f"- \"{s}\"\n")
