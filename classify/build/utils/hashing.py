#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib

def short_hash(text: str, length: int = 8) -> str:
    """Generate a short deterministic SHA1-based hash."""
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:length]
