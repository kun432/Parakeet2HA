import spacy
from spacy.cli import download
import logging
from typing import List, Dict

# Import our validated configuration
from core.config_loader import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NLP_Grammar")

# Map standard language codes to spaCy's small, efficient web models
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_web_sm",
    "fr": "fr_core_web_sm",
    "de": "de_core_web_sm",
    "it": "it_core_web_sm",
    "nl": "nl_core_news_sm",
    "pt": "pt_core_news_sm"
}

class GrammarGenerator:
    """Handles NLP processing to build context-biased grammar for Parakeet."""
    
    def __init__(self):
        self.language_code = config.language
        self.model_name = SPACY_MODELS.get(self.language_code, "en_core_web_sm") # Default to English
        self.nlp = self._ensure_spacy_model()
        
        # Dynamically load wake words from config
        self.wake_words = config.wake_words
        
        # Basic hardcoded verbs for testing (before we implement the HA translation pull)
        self.action_verbs = ["turn on", "turn off", "open", "close", "set", "dim", "toggle"]
        
        # Filler words to help the ASR bridge gaps naturally
        self.fillers = ["the", "my", "in", "to", "please", "could", "you"]

    def _ensure_spacy_model(self):
        """Checks if the language model is installed; if not, downloads it automatically."""
        try:
            logger.info(f"Attempting to load spaCy model '{self.model_name}'...")
            return spacy.load(self.model_name)
        except OSError:
            logger.warning(f"spaCy model '{self.model_name}' not found locally. Downloading now...")
            try:
                download(self.model_name)
                logger.info("Download complete. Loading model...")
                return spacy.load(self.model_name)
            except Exception as e:
                logger.error(f"Failed to auto-download the spaCy model: {e}")
                logger.error(f"Please run manually in your terminal: python -m spacy download {self.model_name}")
                raise SystemExit(1)

    def generate_boosting_list(self, ha_roster: List[Dict]) -> List[str]:
        """
        Takes the clean entity roster from Home Assistant and generates a 
        flat list of highly probable phrases for Parakeet's beam search.
        """
        logger.info("Generating Parakeet grammar boosting list...")
        boosting_phrases = set()
        
        # 1. Add our wake words
        for w in self.wake_words:
            boosting_phrases.add(w.lower())
            
        # 2. Add our basic verbs and fillers
        for v in self.action_verbs:
            boosting_phrases.add(v)
        for f in self.fillers:
            boosting_phrases.add(f)
            
        # 3. Process the Home Assistant Entities
        for item in ha_roster:
            # We want to boost the exact entity name
            if item.get("name"):
                boosting_phrases.add(item["name"])
                
                # We can also pre-compute common combinations to bias the ASR 
                # towards recognizing the whole phrase together
                for verb in self.action_verbs:
                    boosting_phrases.add(f"{verb} the {item['name']}")
            
            # Boost any custom voice aliases the user set
            for alias in item.get("aliases", []):
                boosting_phrases.add(alias)
                for verb in self.action_verbs:
                    boosting_phrases.add(f"{verb} the {alias}")
                    
            # Boost the room name if it exists
            if item.get("room"):
                boosting_phrases.add(item["room"])
                boosting_phrases.add(f"in the {item['room']}")
                
        final_list = list(boosting_phrases)
        logger.info(f"Generated {len(final_list)} context-boosting phrases for ASR.")
        
        return final_list

# =============================================================================
# EXECUTABLE TEST
# =============================================================================
if __name__ == "__main__":
    generator = GrammarGenerator()
    
    # Simulate a tiny roster pulled from HA
    dummy_roster = [
        {"entity_id": "light.kitchen_main", "name": "kitchen light", "aliases": ["big light"], "room": "kitchen"},
        {"entity_id": "switch.coffee_maker", "name": "coffee maker", "aliases": [], "room": "kitchen"}
    ]
    
    phrases = generator.generate_boosting_list(dummy_roster)
    
    print("\n--- SAMPLE BOOSTING PHRASES ---")
    import random
    for p in random.sample(phrases, min(10, len(phrases))):
        print(f"- {p}")
    print("-------------------------------\n")
