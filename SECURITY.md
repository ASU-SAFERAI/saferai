# Security Policy

## Reporting a vulnerability

**Do not open a public issue for a security report.** Report it privately through GitHub: **Security → Advisories → [Report a vulnerability](https://github.com/ASU-SAFERAI/saferai/security/advisories/new)**.

Tell us which component (`pre-deploy`, `post-deploy`, or `in-production`), the commit you tested, how to reproduce it, and the impact of successful exploitation.

Report against the latest commit on the default branch. The components version independently and we do not backport fixes.


## Scope

**A missed detection is not a vulnerability.**
- Prompt-Guard scoring a jailbreak low, Presidio missing a PII string, a keyword or zero-shot label that does not fire, etc. These are accuracy issues. Please open a normal issue with the input that slipped through.

**A defect in the code around the detector is a vulnerability.**
- Anything that crashes or executes code instead of scoring, skips scoring entirely, or exposes credentials or scored content beyond what the configuration asks for.

Out of scope: authentication, TLS, and rate limiting, which is designed to run behind an organization's authenticating API Gateway; whatever service you wrap `pre-deploy` or `post-deploy` in; your own configuration and IAM; and dependency CVEs from a scanner, which belong in a public issue.

If you are unsure, report it privately and we will sort it out.
