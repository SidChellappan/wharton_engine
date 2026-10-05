"""Wharton investment-pipeline engine (diagram boxes 1-10).

The engine produces evidence for a human decision. It never approves a policy,
places a trade, or writes to the WInS competition account.
"""

__version__ = "0.1.0"

ASSETS = ("equity", "treasury", "cash")
EQUITY, TREASURY, CASH = 0, 1, 2
HUMAN_HANDOFF_STATUS = "AWAITING_TEAM_AND_ADVISOR_DECISION"
