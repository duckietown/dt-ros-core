from enum import Enum

# Intersection types
class IntersectionType(Enum):
    Unknown = 0
    StopSign = 1
    TrafficLight = 2


# Actions states to handle overall behavior
class ActionState(Enum):
    Solving = 0   # State when still solving the intersection priority/TL
    Go = 1        # State where we finally publish the GO signal and turn off LEDs
    TimedOut = 2  # When we take too much time and no GO was given


# States for TL processing
class TrafficLightState(Enum):
    Sensing = 0   # State where TL is still being decoded
    Green = 1     # State "green" (flashing) light is read
