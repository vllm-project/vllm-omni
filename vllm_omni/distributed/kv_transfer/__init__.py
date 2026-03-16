"""Patched KV transfer connectors for PD disaggregation.

This package provides monkey-patched versions of vLLM's native KV transfer
connectors (e.g. MooncakeConnector) that fix the request-ID mismatch problem
in prefill-decode disaggregation.

vLLM's ``InputProcessor.assign_request_id()`` appends a random 8-char suffix
to each request ID internally.  The prefill engine stores KV under its own
suffix, but the decode engine generates a *different* suffix — so it can never
find the KV data.  The patched connector threads the prefill engine's internal
``remote_request_id`` through ``kv_transfer_params`` so the decode side can
reference the correct KV entry.
"""
