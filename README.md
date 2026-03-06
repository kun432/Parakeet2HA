# Parakeet2HA
HA voice control 

Just a proof of concept no LLM needed voice control of HA, 1st run wait for parakeet to download and install.
run `python simulate_boww.py` and in `python main.py` in seperate terminals to test maybe put a AGC on your mic [Vlevel
](https://github.com/radiocicletta/vlevel)
The LLM needs for turning on a lightbulb or static fixed strings of HASSIL have always been a confusion for me.
So knocked this up over the last couple of days, its running on a I3-9100 mini pc with HA.
Uses the websockets API so can be situated anywhere and used Parakeet as I am a lazy Dev.
Rather than have hardcoded language strings when HA has translated presentation layers, why not use them.
So Parakeet do to its relatively low compute and fast speed and language support was used a demo.
Same methods with smaller models will work, but you have have individual language models and also maybe implement noise supression as parakeet is very tolerant.

Its extremely fast to update from end of voice command even on a I3-9100.
I have played about with the intent parser enough to get much working, but I don't really create product, just highlight methods and concept.
It can be set as an authoritive 2nd stage wakeword or without and could even replace the need for wakeword if you so wanted.
config.yaml is fairly small and easy to configure and sort of self explanitory.

Anyway have a play, its MIT so copy and molest at your leisure.

if you want to test you can add this to /homeassistant/configuration.yaml via the file editor addon
```
sudo apt-get install -y build-essential cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev git portaudio19-dev
pip install https://kheafield.com/code/kenlm.tar.gz
# or
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir -p build
cd build
cmake ..
make -j $(nproc)
sudo cp bin/lmplz /usr/local/bin/
sudo cp bin/build_binary /usr/local/bin/
lmplz --help
build_binary --help
```

```
# ==============================================================================
# 1. INPUT BOOLEANS & NUMBERS (VIRTUAL HARDWARE STATE)
# ==============================================================================
input_boolean:
  v_light_ceiling:
    name: Virtual Ceiling Light State
  v_light_lamp:
    name: Virtual Table Lamp State
  v_switch_coffee:
    name: Virtual Coffee Maker State
  v_switch_tv:
    name: Virtual TV Plug State
  v_fan_ceiling:
    name: Virtual Ceiling Fan State
  v_fan_desk:
    name: Virtual Desk Fan State
  v_cover_overhead:
    name: Virtual Overhead Door State
  v_lock_primary:
    name: Virtual Primary Door State
    initial: on 
  v_lock_secondary:
    name: Virtual Secondary Door State
    initial: on
  v_bsensor_motion:
    name: Virtual Motion State
  v_bsensor_door:
    name: Virtual Door State
  v_climate_heater:
    name: Virtual Heater Relay State
  v_climate_ac:
    name: Virtual AC Relay State

input_number:
  v_light_ceiling_brightness:
    name: Virtual Ceiling Light Brightness
    min: 0
    max: 255
    step: 1
    initial: 255
  v_fan_ceiling_speed:
    name: Virtual Ceiling Fan Speed
    min: 0
    max: 100
    step: 1
    initial: 50
  v_cover_blinds_pos:
    name: Virtual Window Blinds Position
    min: 0
    max: 100
    step: 1
    initial: 0 
  v_sensor_temp:
    name: Virtual Temperature
    min: 10
    max: 35
    step: 0.5
    initial: 21.0
  v_sensor_humidity:
    name: Virtual Humidity
    min: 0
    max: 100
    step: 1
    initial: 45

# ==============================================================================
# 2. LIGHTS
# ==============================================================================
light:
  - platform: template
    lights:
      virtual_ceiling_light:
        unique_id: virtual_ceiling_light_12345
        friendly_name: "Virtual Ceiling Light"
        value_template: "{{ is_state('input_boolean.v_light_ceiling', 'on') }}"
        level_template: "{{ states('input_number.v_light_ceiling_brightness') | int }}"
        turn_on:
          service: input_boolean.turn_on
          target:
            entity_id: input_boolean.v_light_ceiling
        turn_off:
          service: input_boolean.turn_off
          target:
            entity_id: input_boolean.v_light_ceiling
        set_level:
          - service: input_boolean.turn_on
            target:
              entity_id: input_boolean.v_light_ceiling
          - service: input_number.set_value
            target:
              entity_id: input_number.v_light_ceiling_brightness
            data:
              value: "{{ brightness }}"

      virtual_table_lamp:
        unique_id: virtual_table_lamp_12345
        friendly_name: "Virtual Table Lamp"
        value_template: "{{ is_state('input_boolean.v_light_lamp', 'on') }}"
        turn_on:
          service: input_boolean.turn_on
          target:
            entity_id: input_boolean.v_light_lamp
        turn_off:
          service: input_boolean.turn_off
          target:
            entity_id: input_boolean.v_light_lamp

# ==============================================================================
# 3. SWITCHES
# ==============================================================================
switch:
  - platform: template
    switches:
      virtual_coffee_maker:
        unique_id: virtual_coffee_maker_12345
        friendly_name: "Virtual Coffee Maker Plug"
        value_template: "{{ is_state('input_boolean.v_switch_coffee', 'on') }}"
        turn_on:
          service: input_boolean.turn_on
          target:
            entity_id: input_boolean.v_switch_coffee
        turn_off:
          service: input_boolean.turn_off
          target:
            entity_id: input_boolean.v_switch_coffee

      virtual_tv_plug:
        unique_id: virtual_tv_plug_12345
        friendly_name: "Virtual TV Plug"
        value_template: "{{ is_state('input_boolean.v_switch_tv', 'on') }}"
        turn_on:
          service: input_boolean.turn_on
          target:
            entity_id: input_boolean.v_switch_tv
        turn_off:
          service: input_boolean.turn_off
          target:
            entity_id: input_boolean.v_switch_tv

      virtual_heater_relay:
        unique_id: virtual_heater_relay_12345
        friendly_name: "Virtual Heater"
        value_template: "{{ is_state('input_boolean.v_climate_heater', 'on') }}"
        turn_on:
          service: input_boolean.turn_on
          target:
            entity_id: input_boolean.v_climate_heater
        turn_off:
          service: input_boolean.turn_off
          target:
            entity_id: input_boolean.v_climate_heater

      virtual_ac_relay:
        unique_id: virtual_ac_relay_12345
        friendly_name: "Virtual AC"
        value_template: "{{ is_state('input_boolean.v_climate_ac', 'on') }}"
        turn_on:
          service: input_boolean.turn_on
          target:
            entity_id: input_boolean.v_climate_ac
        turn_off:
          service: input_boolean.turn_off
          target:
            entity_id: input_boolean.v_climate_ac

# ==============================================================================
# 4. FANS
# ==============================================================================
fan:
  - platform: template
    fans:
      virtual_ceiling_fan:
        unique_id: virtual_ceiling_fan_12345
        friendly_name: "Virtual Ceiling Fan"
        value_template: "{{ is_state('input_boolean.v_fan_ceiling', 'on') }}"
        percentage_template: "{{ states('input_number.v_fan_ceiling_speed') | int }}"
        turn_on:
          service: input_boolean.turn_on
          target:
            entity_id: input_boolean.v_fan_ceiling
        turn_off:
          service: input_boolean.turn_off
          target:
            entity_id: input_boolean.v_fan_ceiling
        set_percentage:
          - service: input_boolean.turn_on
            target:
              entity_id: input_boolean.v_fan_ceiling
          - service: input_number.set_value
            target:
              entity_id: input_number.v_fan_ceiling_speed
            data:
              value: "{{ percentage }}"

      virtual_desk_fan:
        unique_id: virtual_desk_fan_12345
        friendly_name: "Virtual Desk Fan"
        value_template: "{{ is_state('input_boolean.v_fan_desk', 'on') }}"
        turn_on:
          service: input_boolean.turn_on
          target:
            entity_id: input_boolean.v_fan_desk
        turn_off:
          service: input_boolean.turn_off
          target:
            entity_id: input_boolean.v_fan_desk

# ==============================================================================
# 5. COVERS
# ==============================================================================
cover:
  - platform: template
    covers:
      virtual_overhead_door:
        unique_id: virtual_overhead_door_12345
        friendly_name: "Virtual Overhead Door"
        device_class: garage
        value_template: "{{ is_state('input_boolean.v_cover_overhead', 'on') }}"
        open_cover:
          service: input_boolean.turn_on
          target:
            entity_id: input_boolean.v_cover_overhead
        close_cover:
          service: input_boolean.turn_off
          target:
            entity_id: input_boolean.v_cover_overhead

      virtual_window_blinds:
        unique_id: virtual_window_blinds_12345
        friendly_name: "Virtual Window Blinds"
        device_class: blind
        position_template: "{{ states('input_number.v_cover_blinds_pos') | int }}"
        set_cover_position:
          service: input_number.set_value
          target:
            entity_id: input_number.v_cover_blinds_pos
          data:
            value: "{{ position }}"

# ==============================================================================
# 6. LOCKS
# ==============================================================================
lock:
  - platform: template
    name: "Virtual Primary Lock"
    unique_id: virtual_primary_lock_12345
    value_template: "{{ is_state('input_boolean.v_lock_primary', 'on') }}"
    lock:
      service: input_boolean.turn_on
      target:
        entity_id: input_boolean.v_lock_primary
    unlock:
      service: input_boolean.turn_off
      target:
        entity_id: input_boolean.v_lock_primary

  - platform: template
    name: "Virtual Secondary Lock"
    unique_id: virtual_secondary_lock_12345
    value_template: "{{ is_state('input_boolean.v_lock_secondary', 'on') }}"
    lock:
      service: input_boolean.turn_on
      target:
        entity_id: input_boolean.v_lock_secondary
    unlock:
      service: input_boolean.turn_off
      target:
        entity_id: input_boolean.v_lock_secondary

# ==============================================================================
# 7. SENSORS & BINARY SENSORS
# ==============================================================================
template:
  - sensor:
      - name: "Virtual Temperature"
        unique_id: virtual_temperature_12345
        unit_of_measurement: "°C"
        device_class: temperature
        state: "{{ states('input_number.v_sensor_temp') }}"
      - name: "Virtual Humidity"
        unique_id: virtual_humidity_12345
        unit_of_measurement: "%"
        device_class: humidity
        state: "{{ states('input_number.v_sensor_humidity') }}"

  - binary_sensor:
      - name: "Virtual Motion"
        unique_id: virtual_motion_12345
        device_class: motion
        state: "{{ is_state('input_boolean.v_bsensor_motion', 'on') }}"
      - name: "Virtual Contact"
        unique_id: virtual_contact_12345
        device_class: door
        state: "{{ is_state('input_boolean.v_bsensor_door', 'on') }}"

# ==============================================================================
# 8. CLIMATE
# ==============================================================================
climate:
  - platform: generic_thermostat
    name: Virtual Thermostat
    unique_id: virtual_thermostat_12345
    heater: switch.virtual_heater_relay
    target_sensor: sensor.virtual_temperature
    min_temp: 15
    max_temp: 30
    ac_mode: false
    target_temp: 22

  - platform: generic_thermostat
    name: Virtual AC Thermostat
    unique_id: virtual_ac_thermostat_12345
    heater: switch.virtual_ac_relay
    target_sensor: sensor.virtual_temperature 
    min_temp: 16
    max_temp: 28
    ac_mode: true
    target_temp: 20
```

Alocate to rooms and add names.
I should do more about just selecting names and domain functions, but as you can see the base logic is there and guess maybe someone will. 

```
# ~/.asoundrc

# Define the virtual 'mic' device
pcm.mic {
    type plug
    slave.pcm "hw:0,0"  # Replace this with your actual loopback hardware ID if different
}

# Optional: You can explicitly hint the rate and channels, 
# though 'plug' will automatically adapt to PyAudio's 16000Hz request.
```
