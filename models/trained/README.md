# models/trained

Models you train on this computer with the Train tab. Nothing here is downloaded
or shipped with Audtheia, and nothing here is committed to the repository, because
a trained model is built from your own field records.

Each training run works in its own folder under runs, which holds its
configuration, its progress, its log, its checkpoints for resuming and the
exported model. Nothing outside that folder is written while it trains.

Every model gets its own folder named Audtheia-Name-V1, Audtheia-Name-V2 and so
on, where Name is the word you chose when you started the training. A finished
run becomes a model when you add it in the Models section of the Train tab. A new
fine-tune of the same name becomes the next version and never overwrites an
earlier one, so you can always go back. Inside a model folder you will find the
model file the application loads, the list of classes it knows, the PyTorch
checkpoint of an image model or the trained layer and calibration of a sound
model, metrics.json with the held-out test figures, and MODEL_CARD.md.

A trained model is a model output, not a measurement. Everything it produces is
stored with the model that produced it, so a report can always say which model
made a result and so measured data and model output stay separate. Switching a
model on is saved in config/settings.local.json and never changes a committed
file.

Training can run for many hours on a computer without a dedicated graphics card.
The Train tab offers a gentle mode that keeps the computer cool and responsive,
and will not start on battery power. Deleting a folder here removes that model
from your computer; switch it off first, with Put back the previous model, if it
is in use.

The training libraries are installed on request into a separate folder, set by
training.paths.envs_dir in the settings, so the field station and the everyday
application never carry them. See docs/training.md for the whole workflow.
