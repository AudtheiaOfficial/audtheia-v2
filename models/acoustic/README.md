# models/acoustic

Swappable acoustic detection models. Which one is active is a setting, so changing
it requires no code change.

- birdnet: the full BirdNET-Analyzer model (for example BirdNET_GLOBAL_6K.tflite),
  the default for terrestrial and coastal bird sites. It runs on the field
  station's CPU.
- marine: the slot for an underwater passive-acoustic model (whales and dolphins,
  fish choruses, snapping-shrimp soundscape indices). BirdNET listens for in-air
  birds and is not suitable for a submerged hydrophone, so a marine site supplies
  its own model here.
- custom: a classifier you fine-tune for your own local or regional species.
