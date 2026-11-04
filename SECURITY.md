# Security Policy

## Reporting a Vulnerability

The Oumi team takes security bugs in Oumi Weights Lab seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

To report a security issue, please use the GitHub Security Advisory ["Report a Vulnerability"](https://github.com/oumi-ai/weights-lab/security/advisories/new) tab. Please don't open a public issue.

The Oumi team will send a response indicating the next steps in handling your report. After the initial reply to your report, the team will keep you informed of the progress towards a fix and full announcement, and may ask for additional information or guidance.

Report security bugs in third-party packages, such as PyTorch or Transformers, to the people who maintain them.

## Scope

The lab runs the Python snippets you or your agent send as trusted code, with your filesystem and network access, as the README says. That is by design. Please do report any way the lab runs code that nobody sent it, for example from a downloaded model, a saved edit bundle, or a session directory.

## Acknowledgements

We deeply appreciate security researchers who take the time to report vulnerabilities to us. We will acknowledge your contribution in our security advisory and release notes (unless you prefer to remain anonymous).

## Security Updates

Security updates and advisories will be published through GitHub Security Advisories.
