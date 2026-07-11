from __future__ import annotations


def unsupported_candidate_adapter(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError("DiffusionDriveV2 does not expose GRPO/candidate scoring adapters yet")


__all__ = ["unsupported_candidate_adapter"]
