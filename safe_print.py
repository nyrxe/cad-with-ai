#!/usr/bin/env python3
"""
Safe printing utility for handling Unicode characters
"""

import sys
import os

def configure_unicode_support():
    """Configure stdout/stderr for Unicode support"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

def safe_print(msg, end='\n'):
    """Print message with Unicode fallback"""
    try:
        print(msg, end=end)
    except UnicodeEncodeError:
        # Replace common Unicode characters with ASCII equivalents
        ascii_msg = msg.replace('✅', '[OK]')
        ascii_msg = ascii_msg.replace('❌', '[X]')
        ascii_msg = ascii_msg.replace('⚠️', '[!]')
        ascii_msg = ascii_msg.replace('🔄', '[>]')
        ascii_msg = ascii_msg.replace('🎯', '[T]')
        ascii_msg = ascii_msg.replace('📊', '[#]')
        ascii_msg = ascii_msg.replace('💡', '[I]')
        ascii_msg = ascii_msg.replace('🚀', '[R]')
        ascii_msg = ascii_msg.replace('🧹', '[C]')
        ascii_msg = ascii_msg.replace('→', '->')
        ascii_msg = ascii_msg.replace('←', '<-')
        ascii_msg = ascii_msg.replace('↑', '^')
        ascii_msg = ascii_msg.replace('↓', 'v')
        
        # Remove any remaining non-ASCII characters
        ascii_msg = ''.join(char if ord(char) < 128 else '?' for char in ascii_msg)
        print(ascii_msg, end=end)

# Configure Unicode support when imported
configure_unicode_support()
