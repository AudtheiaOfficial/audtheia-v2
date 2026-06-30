# tests

Unit, integration, and mocked-hardware tests for the system.

The software is designed to be built and verified entirely on mocked hardware, so
no Raspberry Pi, accelerator, camera, or hydrophone is required to run the tests.
Each module's verification graduates into this folder as that module is built, so
the checks live with the code rather than only in development notes.
