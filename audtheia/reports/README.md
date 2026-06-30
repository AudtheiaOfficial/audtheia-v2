# audtheia/reports

Report generation, on the desktop hub.

| File | Runs on | Role |
|------|---------|------|
| generate.py | Desktop hub | Produces PDF and CSV reports from the verified record and the longitudinal analysis. Every value is labeled with its source and its quality-control status, and discovered patterns are presented as candidate hypotheses with an effect size and a data span, never as established findings. |

Reporting is one of only two activities in the system that run on a schedule you
set (daily, weekly, biweekly, or on demand). The other is the longitudinal pass.
