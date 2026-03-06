import os
import sys
import yaml

class AppConfig:
    """Loads and validates the external YAML configuration file."""
    
    def __init__(self, config_path="config.yaml"):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(base_dir, config_path)
        
        self.data = self._load_yaml()
        self._validate_config()
        
        # --- Connection Variables ---
        self.ha_host = self.data['home_assistant']['host']
        self.ha_port = self.data['home_assistant']['port']
        self.use_ssl = self.data['home_assistant']['use_ssl']
        self.master_token = self.data['home_assistant']['master_token']
        self.language = self.data['home_assistant']['default_language']
        
        protocol = "wss" if self.use_ssl else "ws"
        self.ws_uri = f"{protocol}://{self.ha_host}:{self.ha_port}/api/websocket"
        
        # --- ASR Variables ---
        self.asr_model = self.data['asr_settings']['model_name']
        self.silence_dur = self.data['asr_settings']['silence_duration']
        
        self.process_interval = self.data['asr_settings'].get('process_interval', 0.5)
        self.stability_limit = self.data['asr_settings'].get('stability_limit', 3)
        
        self.input_device_name = self.data['asr_settings'].get('input_device_name', 'mic')
        
        # --- Wake Word Logic ---
        self.wake_words = self.data['asr_settings'].get('wake_words', ["hey jarvis", "jarvis"])
        self.ww_authoritative = self.data['asr_settings'].get('wake_word_authoritative', False)
        self.ww_strictness = self.data['asr_settings'].get('wake_word_strictness', 0.80)
        
        self.show_vu_meter = self.data['asr_settings'].get('show_vu_meter', True)
        
        # --- NLP Aliases ---
        self.domain_aliases = self.data.get('domain_aliases', {})
        
        self.users = self.data.get('users', [])

    def _load_yaml(self):
        if not os.path.exists(self.config_path):
            print(f"[ERROR] Could not find '{self.config_path}'. Please create it in the root directory.")
            sys.exit(1)
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"[ERROR] YAML syntax error in '{self.config_path}': {exc}")
            sys.exit(1)

    def _validate_config(self):
        required_ha_keys = ['host', 'port', 'master_token', 'default_language']
        
        if 'home_assistant' not in self.data:
            print("[ERROR] Missing 'home_assistant' block in config.yaml")
            sys.exit(1)
            
        for key in required_ha_keys:
            if key not in self.data['home_assistant'] or not self.data['home_assistant'][key]:
                print(f"[ERROR] Missing required configuration: home_assistant -> {key}")
                sys.exit(1)

    def get_user_token(self, username):
        for user in self.users:
            if user['name'].lower() == username.lower():
                return user.get('token')
        return self.master_token

config = AppConfig()
