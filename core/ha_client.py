import json
import logging
import asyncio
import websockets
from core.config_loader import config

logger = logging.getLogger("HA_Client")

class HomeAssistantClient:
    """Handles the WebSocket connection and registry mapping for Home Assistant."""
    
    def __init__(self):
        self.uri = config.ws_uri
        self.token = config.master_token
        self.language = getattr(config, 'language', 'en')
        self.ws = None
        self.message_id = 1
        
        self.voice_roster = []
        self.translations = {}
        self.domain_services = {}
        
        # Async dictionary to map message IDs to their response futures
        self._pending_requests = {}

    async def _receive_loop(self):
        """Listens continuously for responses from Home Assistant."""
        try:
            async for message in self.ws:
                data = json.loads(message)
                if "id" in data and data["id"] in self._pending_requests:
                    self._pending_requests[data["id"]].set_result(data)
        except Exception as e:
            logger.error(f"WebSocket receive loop closed or errored: {e}")

    async def _send_request(self, payload: dict) -> dict:
        """Packages a payload with a message ID and waits for the specific response."""
        if not self.ws:
            return {}
        
        payload["id"] = self.message_id
        msg_id = self.message_id
        self.message_id += 1
        
        future = asyncio.get_running_loop().create_future()
        self._pending_requests[msg_id] = future
        
        await self.ws.send(json.dumps(payload))
        
        try:
            # Wait up to 5 seconds for HA to reply
            response = await asyncio.wait_for(future, timeout=5.0)
            return response
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for HA response to request {msg_id}")
            return {}
        finally:
            self._pending_requests.pop(msg_id, None)

    async def bootstrap_system(self) -> bool:
        """Connects to HA, authenticates, and downloads the spatial and language context."""
        logger.info(f"Connecting to Home Assistant at {self.uri}...")
        try:
            self.ws = await websockets.connect(self.uri)
            await self.ws.recv() # Discard auth_required
            
            await self.ws.send(json.dumps({"type": "auth", "access_token": self.token}))
            auth_result = json.loads(await self.ws.recv())
            
            if auth_result.get("type") != "auth_ok":
                logger.error("Home Assistant authentication failed.")
                return False
                
            logger.info("🟢 Authenticated with Home Assistant.")
            asyncio.create_task(self._receive_loop())

            # 1. Fetch Registries (Areas, Devices, Entities)
            logger.info("Fetching Area, Device, and Entity Registries...")
            areas_resp = await self._send_request({"type": "config/area_registry/list"})
            area_map = {a["area_id"]: a["name"] for a in areas_resp.get("result", []) if "area_id" in a}

            devices_resp = await self._send_request({"type": "config/device_registry/list"})
            device_map = {d["id"]: d.get("area_id") for d in devices_resp.get("result", []) if "id" in d}

            entities_resp = await self._send_request({"type": "config/entity_registry/list"})
            entity_registry = {e["entity_id"]: e for e in entities_resp.get("result", []) if "entity_id" in e}

            # 2. Fetch Live States
            logger.info("Fetching States...")
            states_resp = await self._send_request({"type": "get_states"})
            states = states_resp.get("result", [])

            # 3. Fetch Localization/Translations
            logger.info(f"Fetching '{self.language}' translations...")
            trans_resp = await self._send_request({
                "type": "frontend/get_translations",
                "language": self.language,
                "category": "services"
            })
            self.translations = trans_resp.get("result", {}).get("resources", {})

            # 4. Fetch the Master Service Blueprint
            logger.info("Fetching Domain Service Blueprint...")
            services_resp = await self._send_request({"type": "get_services"})
            raw_services = services_resp.get("result", {})
            
            self.domain_services = {domain: list(srvs.keys()) for domain, srvs in raw_services.items()}

            # 5. Build Spatially-Aware Roster
            self.voice_roster = []
            for state in states:
                ent_id = state["entity_id"]
                domain = ent_id.split(".")[0]
                reg_info = entity_registry.get(ent_id, {})
                
                area_id = reg_info.get("area_id")
                if not area_id:
                    device_id = reg_info.get("device_id")
                    if device_id:
                        area_id = device_map.get(device_id)
                        
                area_name = area_map.get(area_id, "")

                self.voice_roster.append({
                    "entity_id": ent_id,
                    "domain": domain,
                    "name": state.get("attributes", {}).get("friendly_name", ""),
                    "area_id": area_id,  # <--- THE MISSING LINK IS NOW HERE
                    "area_name": area_name
                })

            logger.info(f"🟢 Built voice roster with {len(self.voice_roster)} entities.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to bootstrap HA: {e}")
            return False

    async def execute_service(self, domain: str, service: str, entity_id: str = None, area_id: str = None, payload: dict = None) -> bool:
        """Fires a service call over the WebSocket."""
        if not payload:
            payload = {}
            
        target_dict = {}
        if entity_id:
            target_dict["entity_id"] = entity_id
        elif area_id:
            target_dict["area_id"] = area_id
            
        service_data = {
            "type": "call_service",
            "domain": domain,
            "service": service,
            "target": target_dict,
            "service_data": payload
        }
        
        logger.info(f"Executing HA Service: {domain}.{service} | Target: {target_dict}")
        response = await self._send_request(service_data)
        return response.get("success", False)
