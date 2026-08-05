# Security Policy

## Reporting a vulnerability

If you believe you have found a security vulnerability in Audtheia V2, please
report it privately rather than opening a public issue. Use GitHub's private
vulnerability reporting for this repository (the **Security** tab, "Report a
vulnerability") so the report is not visible while it is being addressed.

Please include the affected file or component, the platform you were running on
(the desktop hub on Windows, macOS, or Linux, or a Raspberry Pi field station),
the steps to reproduce, and the impact you observed. You can expect an initial
acknowledgement within a reasonable time, and an update once the report has been
assessed.

## Threat model

Audtheia V2 is a local application. It is important to understand what that means
for security.

- **No runtime cloud.** After setup, the platform runs entirely on hardware you
  own and does not send observations to any external service. The network is used
  only to reach your own field stations and, at setup time, to fetch species
  reference data from public scientific databases under your own credentials.
- **What is sensitive.** The most sensitive assets are the local scientific
  record (the SQLite database and the captured media under `data/`, `database/`,
  `reports/`, and `exports/`, all of which are excluded from version control) and
  any credentials you provide. Only one credential changes behavior: an IUCN Red
  List token, stored in `config/secrets.json`, which is git-ignored.
- **Trust boundary.** The desktop server binds to the loopback address and is
  intended for use on a trusted machine and, for field stations, a trusted local
  network. It is not hardened for exposure to the public internet, and it should
  not be exposed to it.

## Handling of credentials and personal data

Credentials live in git-ignored files (`config/secrets.json`, `*.env`) and are
never committed. Absolute machine paths typed into the interface are relocated to
a git-ignored local overrides file rather than the shared, tracked configuration,
so ordinary use does not leak a local path into version control.

## Supported versions

Audtheia V2 is under active development. Security fixes are applied to the latest
released version on the `main` branch. Older snapshots are not maintained.
