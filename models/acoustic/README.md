# models/acoustic

Swappable acoustic detection models. Which one is active is a setting, so changing it requires no code change. The acoustic model runs on the field station's processor, and an acoustic detection can open a full observation on its own, which is also what keeps a station recording when a camera view degrades. Model files are not committed to the repository because of their size; place them here or let setup stage them.

- **birdnet**: the full BirdNET-Analyzer model (for example `BirdNET_GLOBAL_6K.tflite`), the default for terrestrial and coastal bird sites. It runs on the field station's processor. BirdNET is licensed CC BY-NC-SA 4.0, which permits non-commercial use with attribution and share-alike, so keep that in mind for commercial work.
- **marine**: the slot for an underwater passive-acoustic model (whales and dolphins, fish choruses, snapping-shrimp soundscape indices). BirdNET is trained on in-air bird sound and is not suitable for a submerged hydrophone, so a marine site supplies its own model here.
- **custom**: a classifier you fine-tune for your own local or regional species. Collect local recordings, organize them into one folder per species, run BirdNET's training, and drop the result here, then select it in the settings. See [docs/custom-models.md](../../docs/custom-models.md).
