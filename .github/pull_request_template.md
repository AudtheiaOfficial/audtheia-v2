<!-- Thank you for contributing to Audtheia. Please keep this short and honest. -->

**What this changes**
A clear description of the change and why.

**Design rules kept** (see [CONTRIBUTING.md](../CONTRIBUTING.md))
- [ ] Measured, referenced, inferred, and pattern-derived values stay distinguishable; nothing blurs a measurement with an inference.
- [ ] No model family, species, credential, or absolute path is hardcoded; configuration stays in `settings.json` and secrets stay in git-ignored files.
- [ ] No personal information, no em dashes, and no emojis in tracked files.

**Tests**
- [ ] `python tests/run_all.py` passes locally.
- [ ] New behavior is covered by a test where practical.

**Notes**
Anything a reviewer should know, including on-device validation status if this
touches the field drivers.
