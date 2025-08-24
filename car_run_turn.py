import RPi.GPIO as GPIO
import time


Motor_R1_Pin = 16
Motor_R2_Pin = 18
Motor_L1_Pin = 11
Motor_L2_Pin = 13
DEFAULT_STEP_S = 1.0  # default movement duration (seconds)


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


def forward(seconds: float = DEFAULT_STEP_S):
    move(r_forward=True, l_forward=True, seconds=seconds)


def backward(seconds: float = DEFAULT_STEP_S):
    move(r_forward=False, l_forward=False, seconds=seconds)


def turnRight(seconds: float = DEFAULT_STEP_S):
    # Right wheel backward or stop, left wheel forward to pivot
    GPIO.output(Motor_R1_Pin, GPIO.LOW)
    GPIO.output(Motor_R2_Pin, GPIO.LOW)
    GPIO.output(Motor_L1_Pin, GPIO.HIGH)
    GPIO.output(Motor_L2_Pin, GPIO.LOW)
    time.sleep(max(0.0, seconds))
    stop()


def turnLeft(seconds: float = DEFAULT_STEP_S):
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
            raw = input('(f [sec], b [sec], r [sec], l [sec], q): ').strip()
            if not raw:
                continue
            parts = raw.split()
            ch = parts[0].lower()
            sec = float(parts[1]) if len(parts) > 1 else DEFAULT_STEP_S

            if ch == 'f':
                forward(sec)
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