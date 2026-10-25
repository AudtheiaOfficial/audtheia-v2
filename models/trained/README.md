# models/trained

Models you train on this computer with the Train tab. Nothing here is downloaded
or shipped with Audtheia, and nothing here is committed to the repository, because
a trained model is built from your own field records.

Every model gets its own folder named Audtheia-Name-V1, Audtheia-Name-V2 and so
on, where Name is the word you chose when you started the training. A new
fine-tune of the same name becomes the next version and never overwrites an
earlier one, so you can always go back. Inside a model folder you will find the
model file the application loads, the list of classes it knows, its saved
training checkpoint when one was kept, and a results file with the held-out test
figures that were measured when it finished.

A trained model is a model output, not a measurement. Everything it produces is
stored with the model that produced it, so a report can always say which model
made a result and so measured data and model output stay separate.

Training can run for many hours on a computer without a dedicated graphics card.
The Train tab therefore defaults to a gentle mode on such a machine, keeps the
computer cool and responsive, and will not start on battery power. Deleting a
folder here removes that model from your computer. A model that is currently
switched on in your settings cannot be removed until you switch back.

The training libraries are installed on request into a separate folder named
.venv-train beside the application, so the field station and the everyday
application never carry them.
