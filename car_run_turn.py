import RPi.GPIO as GPIO
import time


Motor_R1_Pin = 16
Motor_R2_Pin = 18
Motor_L1_Pin = 11
Motor_L2_Pin = 13
DEFAULT_STEP_S = 1.0  # default movement duration (seconds)
DEFAULT_STEP_S_turn = 0.8  # default turn duration (seconds)

# Trim controls left/right bias for straight driving.
# Positive TRIM means the robot tends to drift RIGHT, so we will slightly "weaken" the RIGHT wheel.
# Range recommendation: -0.3 ~ +0.3
TRIM_DEFAULT = 0.0
_SLICE_S = 0.05  # time-slicing for software trim (sec)


GPIO.setmode(GPIO.BOARD)
GPIO.setup(Motor_R1_Pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor_R2_Pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor_L1_Pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Motor_L2_Pin, GPIO.OUT, initial=GPIO.LOW)


def move(r_forward: bool, l_forward: bool, seconds: float):
    """Drive both motors in the given directions for 'seconds' seconds."""
    # Right motor
    GPIO.output(Motor_R1_Pin, GPIO.HIGH if r_forward else GPIO.LOW)
    GPIO.output(Motor_R2_Pin, GPIO.LOW if r_forward else GPIO.HIGH)
    # Left motor
    GPIO.output(Motor_L1_Pin, GPIO.HIGH if l_forward else GPIO.LOW)
    GPIO.output(Motor_L2_Pin, GPIO.LOW if l_forward else GPIO.HIGH)
    time.sleep(max(0.0, seconds))
    stop()


def stop():
    GPIO.output(Motor_R1_Pin, False)
    GPIO.output(Motor_R2_Pin, False)
    GPIO.output(Motor_L1_Pin, False)
    GPIO.output(Motor_L2_Pin, False)


def forward(seconds: float = DEFAULT_STEP_S, trim: float = TRIM_DEFAULT):
    """
    Drive forward for 'seconds' with optional trim.
    trim > 0: robot tends to drift RIGHT -> weaken RIGHT wheel
    trim < 0: robot tends to drift LEFT  -> weaken LEFT  wheel
    """
    # Small trims are handled by time-slicing both wheels and shortening ON-time on the stronger side.
    t_rem = max(0.0, seconds)
    # clamp trim for safety
    if trim > 0.5:
        trim = 0.5
    if trim < -0.5:
        trim = -0.5
    if abs(trim) < 1e-6:
        move(r_forward=True, l_forward=True, seconds=t_rem)
        return
    # pre-configure motor forward states
    r_fwd = (GPIO.HIGH, GPIO.LOW)   # (R1, R2) forward
    l_fwd = (GPIO.HIGH, GPIO.LOW)   # (L1, L2) forward
    r_off = (GPIO.LOW, GPIO.LOW)
    l_off = (GPIO.LOW, GPIO.LOW)
    while t_rem > 1e-6:
        dt = _SLICE_S if t_rem > _SLICE_S else t_rem
        # set both wheels forward
        GPIO.output(Motor_R1_Pin, r_fwd[0]); GPIO.output(Motor_R2_Pin, r_fwd[1])
        GPIO.output(Motor_L1_Pin, l_fwd[0]); GPIO.output(Motor_L2_Pin, l_fwd[1])
        if trim > 0:
            # weaken RIGHT wheel by turning it off for dt*trim within this slice
            on_r = dt * (1.0 - trim)
            if on_r > 0:
                time.sleep(on_r)
            # right wheel off for the rest of the slice, keep left on
            GPIO.output(Motor_R1_Pin, r_off[0]); GPIO.output(Motor_R2_Pin, r_off[1])
            time.sleep(max(0.0, dt - on_r))
        else:
            # weaken LEFT wheel
            on_l = dt * (1.0 + trim)  # trim negative -> reduces on-time
            if on_l > 0:
                time.sleep(on_l)
            GPIO.output(Motor_L1_Pin, l_off[0]); GPIO.output(Motor_L2_Pin, l_off[1])
            time.sleep(max(0.0, dt - on_l))
        t_rem -= dt
    stop()


def backward(seconds: float = DEFAULT_STEP_S):
    move(r_forward=False, l_forward=False, seconds=seconds)


def turnRight(seconds: float = DEFAULT_STEP_S_turn):
    # Right wheel backward or stop, left wheel forward to pivot
    GPIO.output(Motor_R1_Pin, GPIO.LOW)
    GPIO.output(Motor_R2_Pin, GPIO.LOW)
    GPIO.output(Motor_L1_Pin, GPIO.HIGH)
    GPIO.output(Motor_L2_Pin, GPIO.LOW)
    time.sleep(max(0.0, seconds))
    stop()


def turnLeft(seconds: float = DEFAULT_STEP_S_turn):
    # Left wheel backward or stop, right wheel forward to pivot
    GPIO.output(Motor_R1_Pin, GPIO.HIGH)
    GPIO.output(Motor_R2_Pin, GPIO.LOW)
    GPIO.output(Motor_L1_Pin, GPIO.LOW)
    GPIO.output(Motor_L2_Pin, GPIO.LOW)
    time.sleep(max(0.0, seconds))
    stop()


if __name__ == "__main__":

    try:
        while True:
            raw = input('(f [sec] [trim], b [sec], r [sec], l [sec], q)  trim>0 weakens RIGHT wheel (corrects right drift): ').strip()
            if not raw:
                continue
            parts = raw.split()
            ch = parts[0].lower()
            sec = float(parts[1]) if len(parts) > 1 else DEFAULT_STEP_S
            trim = float(parts[2]) if (ch == 'f' and len(parts) > 2) else TRIM_DEFAULT

            if ch == 'f':
                forward(sec, trim)
            elif ch == 'b':
                backward(sec)
            elif ch == 'r':
                turnRight(sec)
            elif ch == 'l':
                turnLeft(sec)
            elif ch == 'q':
                print("\nQuit")
                break
            else:
                print("Unknown command. Use f/b/r/l/q")
    finally:
        GPIO.cleanup()