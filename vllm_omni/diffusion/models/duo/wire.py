# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tensor/dict wire layouts for official diffusion request preprocessing."""

from dataclasses import asdict

from .codec import DuoCodecBlock, validate_codec_block
from .contracts import RealizationGroup, RealizationSequence
from .pipeline import RealizationBlockCondition, validate_condition


def _sequence_to_wire(sequence):
    if sequence is None:
        return None
    return {
        name: [asdict(g) for g in value] if name in ("groups", "refiner_groups") else value
        for name in sequence.__dataclass_fields__
        for value in (getattr(sequence, name),)
    }


def _sequence_from_wire(value):
    if value is None:
        return None
    fields = dict(value)
    for key in ("groups", "refiner_groups"):
        fields[key] = tuple(RealizationGroup(**g) for g in fields[key])
    return RealizationSequence(**fields).validate()


def condition_to_wire(condition):
    return {
        name: _sequence_to_wire(value) if name in ("sequence", "prefix") else value
        for name in condition.__dataclass_fields__
        for value in (getattr(condition, name),)
    }


def condition_from_wire(value):
    fields = dict(value)
    for name in ("sequence", "prefix"):
        fields[name] = _sequence_from_wire(fields[name])
    condition = RealizationBlockCondition(**fields)
    validate_condition(condition)
    return condition


def codec_to_wire(block):
    return {name: getattr(block, name) for name in block.__dataclass_fields__}


def codec_from_wire(value):
    block = DuoCodecBlock(**value)
    validate_codec_block(block)
    return block
