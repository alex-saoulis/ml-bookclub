from enum import Enum, IntEnum

class BRACKETS(IntEnum):
    DEALER_STICK = 17,
    BUST_UP = 22,
    BUST_DOWN = 0

class CARD_VALUES(IntEnum):
    MIN = 1,
    MAX = 10

class COLOURS(Enum):
    BLACK = "black",
    RED = "red"

class REWARDS(IntEnum):
    NEGATIVE = -1,
    POSITIVE = 1,
    NEUTRAL = 0



