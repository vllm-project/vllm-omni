# Security Policy

## Supported Versions

The following versions of vLLM-omni are currently being supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of vLLM-omni seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please Do Not

- **Do not** open a public GitHub issue for security vulnerabilities
- **Do not** disclose the vulnerability publicly until it has been addressed

### Please Do

**Report security vulnerabilities privately** via one of these methods:

1. **Preferred**: Use GitHub's private vulnerability reporting feature:
   - Go to the [Security tab](https://github.com/hsliuustc0106/vllm-omni/security) of our repository
   - Click "Report a vulnerability"
   - Fill out the form with details

2. **Email**: Send details to [hsliuustc@gmail.com](mailto:hsliuustc@gmail.com)
   - Use subject line: `[SECURITY] Description of the issue`
   - Include "SECURITY" in the subject

### What to Include

When reporting a vulnerability, please include:

1. **Description**: A clear description of the vulnerability
2. **Impact**: What could an attacker potentially do?
3. **Reproduction**: Step-by-step instructions to reproduce the issue
4. **Environment**: 
   - vLLM-omni version
   - vLLM version
   - Python version
   - Operating system
   - GPU information (if relevant)
5. **Proof of Concept**: Code or commands that demonstrate the vulnerability (if applicable)
6. **Suggested Fix**: Any ideas you have for fixing the issue (optional)

### Example Report

```
Subject: [SECURITY] Arbitrary Code Execution in Model Loading

Description:
A vulnerability exists in the model loading functionality that allows
arbitrary code execution when loading models from untrusted sources.

Impact:
An attacker could craft a malicious model file that executes arbitrary
code when loaded by vLLM-omni.

Reproduction Steps:
1. Create a model file with embedded Python code
2. Load the model using OmniLLM.from_pretrained()
3. The embedded code executes during model initialization

Environment:
- vLLM-omni version: 0.1.0
- vLLM version: 0.10.2
- Python 3.12
- Ubuntu 22.04

Proof of Concept:
[Attached file or code snippet]

Suggested Fix:
Add validation and sandboxing for model loading operations.
```

## Response Process

### Timeline

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 5 business days
- **Status Updates**: We will keep you informed of our progress
- **Resolution**: We aim to address critical vulnerabilities within 30 days

### What Happens Next

1. **Verification**: We will verify the vulnerability and assess its severity
2. **Fix Development**: We will develop and test a fix
3. **Disclosure**: We will coordinate with you on responsible disclosure timing
4. **Release**: We will release a security update
5. **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)

## Security Update Process

When we release security updates:

1. **Security Advisory**: We will publish a GitHub Security Advisory
2. **Release Notes**: Security fixes will be documented in CHANGELOG.md
3. **Notification**: Users will be notified through:
   - GitHub Security Advisories
   - Release notes
   - GitHub Discussions (for critical issues)

## Severity Levels

We use the following severity classifications:

### Critical
- Remote code execution
- Privilege escalation
- Data breach or loss
- Authentication bypass

### High
- Denial of service
- Information disclosure (sensitive data)
- Security feature bypass

### Medium
- Information disclosure (non-sensitive)
- Security misconfiguration
- Insufficient logging

### Low
- Minor information disclosure
- Security improvements
- Best practice violations

## Security Best Practices

### For Users

When using vLLM-omni:

1. **Keep Updated**: Always use the latest version
2. **Trust Sources**: Only load models from trusted sources
3. **Network Security**: Secure your API endpoints properly
4. **Access Control**: Implement proper authentication and authorization
5. **Input Validation**: Validate all user inputs
6. **Resource Limits**: Set appropriate resource limits to prevent DoS
7. **Monitor Logs**: Regularly review application logs for suspicious activity

### For Developers

When contributing to vLLM-omni:

1. **Input Validation**: Always validate and sanitize inputs
2. **Secure Dependencies**: Keep dependencies updated and scan for vulnerabilities
3. **Code Review**: All code should be reviewed for security issues
4. **Testing**: Include security tests in your test suite
5. **Secrets**: Never commit secrets, tokens, or credentials
6. **Error Handling**: Don't expose sensitive information in error messages

## Dependency Security

We monitor dependencies for known vulnerabilities:

- **Regular Updates**: Dependencies are regularly reviewed and updated
- **Automated Scanning**: We use automated tools to detect vulnerable dependencies
- **Version Pinning**: Critical dependencies are pinned to secure versions

## Known Security Considerations

### Model Security

- **Model Provenance**: Only load models from trusted sources
- **Model Scanning**: Consider scanning models for malicious code before loading
- **Sandboxing**: We recommend running model inference in isolated environments

### API Security

- **Authentication**: Implement authentication for API endpoints
- **Rate Limiting**: Use rate limiting to prevent abuse
- **Input Size Limits**: Enforce limits on input sizes
- **HTTPS**: Always use HTTPS in production

### Multi-Modal Inputs

- **Image Processing**: Validate image files before processing
- **Audio Processing**: Validate audio files before processing
- **File Types**: Restrict allowed file types and sizes

## Contact

For security-related questions that are not vulnerabilities:
- Email: hsliuustc@gmail.com
- Subject line: `[SECURITY QUESTION] Your question`

For general questions:
- GitHub Discussions
- GitHub Issues (for non-security bugs)

## Acknowledgments

We would like to thank all security researchers who responsibly disclose vulnerabilities to us. Your efforts help make vLLM-omni more secure for everyone.

### Security Contributors

We will list security researchers who have helped improve vLLM-omni's security here (with permission):

- *Your name could be here!*

---

**Thank you for helping keep vLLM-omni and our users safe!**
