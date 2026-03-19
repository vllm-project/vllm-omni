# CI Failures

What should I do when a CI job fails on my PR, but I don't think my PR caused the failure?


## Common Case 1:

🚨 Error: The command was interrupted by a signal: signal: terminated

Reason: The test is terminated due to exceed the time limits. Sometimes rebuild the test will pass.
