# Security Policy

## Reporting security issues

Please report security issues privately using [the vulnerability submission form](https://github.com/vllm-project/vllm-omni/security/advisories/new).

Do not share vulnerability details in public GitHub issues, pull requests, discussions, or Slack channels. If you cannot use the advisory form, send a direct message to one of the [vLLM-Omni lead maintainers](docs/community/governance.md#lead-maintainers) in [vLLM Slack](https://slack.vllm.ai/). If you cannot send a direct message, you may post a detail-free request in the `sig-omni` channel asking a lead maintainer to contact you privately. Do not include any vulnerability details in that request.

## Issue triage

Reports will be triaged by the vLLM-Omni maintainers and coordinated with the upstream [vLLM vulnerability management team](https://docs.vllm.ai/en/latest/contributing/vulnerability_management.html) when applicable.

## Threat model

vLLM-Omni extends vLLM with omni-modality serving (text, image, audio, video, and action), including diffusion image, video, and audio generation, TTS, and robot-policy models. Please see the [vLLM Security Guide](https://docs.vllm.ai/en/latest/usage/security.html) for information on vLLM's security assumptions and recommendations.

Please see [PyTorch's Security Policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md) for more information and recommendations on how to securely interact with models.

## Issue severity

We will determine the risk of each issue, taking into account our experience dealing with past issues, versions affected, common defaults, and use cases. We use the following severity categories:

### CRITICAL Severity

Vulnerabilities that allow remote attackers to execute arbitrary code, take full control of the system, or significantly compromise confidentiality, integrity, or availability without user interaction or privileges. Examples include remote code execution over the network and deserialization issues that allow exploit chains. These issues are generally rated CVSS ≥ 9.0.

### HIGH Severity

Serious security flaws that allow elevated impact—like RCE in specific, limited contexts or significant data loss—but require advanced conditions or some trust. Examples include RCE in advanced deployment modes (e.g. multi-node), or high impact issues where some sort of privileged network access is required. These issues typically have CVSS scores between 7.0 and 8.9.

### MODERATE Severity

Vulnerabilities that cause denial of service or partial disruption, but do not allow arbitrary code execution or data breach and have limited impact. These issues have a CVSS rating between 4.0 and 6.9.

### LOW Severity

Minor issues such as informational disclosures, logging errors, non-exploitable flaws, or weaknesses that require local or high-privilege access and offer negligible impact. Examples include side channel attacks or hash collisions. These issues often have CVSS scores less than 4.0.

## Fix disclosure policy

When a security report is accepted, the fix process depends on the severity:

* **CRITICAL and HIGH severity**: Fixes are developed in a private security fork and coordinated with the prenotification group before public disclosure.
* **MODERATE and LOW severity**: Fixes are developed and submitted as public pull requests. These issues do not require an embargo, because they do not enable arbitrary code execution or a significant data breach, and public visibility accelerates community review and adoption of the fix.

The [vLLM-Omni lead maintainers](docs/community/governance.md#lead-maintainers) reserve the right to adjust the disclosure approach on a case-by-case basis, taking into account factors such as active exploitation, unusual attack surface, or coordination requirements with downstream vendors.

## CVE assignment and advisory publication

After a security advisory has been accepted:

* A CVE is requested when a patch for the advisory is merged.
* The advisory is published when the fix lands in a vLLM-Omni release.

## Prenotification policy

For security issues of CRITICAL or HIGH severity, we may prenotify organizations or vendors that ship vLLM-Omni. The purpose of this prenotification is to allow a coordinated release of fixes for severe issues.

MODERATE issues are fixed in public pull requests and are not embargoed. A MODERATE issue may still receive a short advance notice when downstream coordination is useful. That notice does not delay the public pull request or hide the remediation.

* This prenotification will be in the form of a private email notification. It may also include adding security contacts to the GitHub security advisory, typically a few days before release.
* If you wish to be added to the prenotification group, please contact the [vLLM-Omni lead maintainers](docs/community/governance.md#lead-maintainers). Each vendor contact will be analyzed on a case-by-case basis. Reach them by direct message in [vLLM Slack](https://slack.vllm.ai/), as described in the [security disclosure instructions](docs/community/contact_us.md#security-disclosures).
* Organizations and vendors that ship or use vLLM-Omni are eligible to join the prenotification group if they meet at least one of the following qualifications:
    * A substantial internal deployment of vLLM-Omni.
    * An established internal security team and comprehensive compliance measures.
    * Active and consistent contributions to vLLM-Omni.
* We may withdraw organizations from receiving future prenotifications if they release fixes or any other information about issues before they are public. Group membership may also change based on policy refinements for who may be included.
