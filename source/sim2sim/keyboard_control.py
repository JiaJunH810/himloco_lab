"""Keyboard velocity command controller for sim2sim scripts.

Provides arrow-key control of linear velocity (vx, vy) with incremental
adjustments. Works with any robot sim2sim script by providing a simple
get_command() interface.

Usage:
    ctrl = KeyboardCommand(vx_step=0.5, vy_step=0.5, max_v=2.0)
    ...
    cmd = ctrl.get_command()  # returns np.array([vx, vy, omega])
"""

import numpy as np

try:
    import glfw
    HAS_GLFW = True
except ImportError:
    HAS_GLFW = False


class KeyboardCommand:
    """Arrow-key velocity command controller.

    Keys:
        Up      → +vx (forward)
        Down    → -vx (backward)
        Left    → +vy (left)
        Right   → -vy (right)
        Space   → reset to zero
        R       → reset to zero

    Args:
        vx_step: Increment per press for forward/backward velocity.
        vy_step: Increment per press for lateral velocity.
        max_v: Maximum absolute velocity magnitude.
    """

    def __init__(self, vx_step: float = 0.5, vy_step: float = 0.5, max_v: float = 2.0):
        self._vx_step = vx_step
        self._vy_step = vy_step
        self._max_v = max_v
        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0

        self._prev_up = False
        self._prev_down = False
        self._prev_left = False
        self._prev_right = False
        self._prev_space = False
        self._prev_r = False

    def update(self, window) -> np.ndarray:
        """Read keyboard state and update velocity command.

        Call this once per control step (not per sim step).

        Args:
            window: The glfw window handle from mujoco_viewer.

        Returns:
            np.ndarray of shape (3,) with [vx, vy, omega].
        """
        if not HAS_GLFW or window is None:
            return np.array([self._vx, self._vy, self._omega], dtype=np.float32)

        up = glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS
        down = glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS
        left = glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS
        right = glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS
        space = glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS
        r = glfw.get_key(window, glfw.KEY_R) == glfw.PRESS

        # Edge-triggered: only act on press (not hold)
        if up and not self._prev_up:
            self._vx += self._vx_step
        if down and not self._prev_down:
            self._vx -= self._vx_step
        if left and not self._prev_left:
            self._vy += self._vy_step
        if right and not self._prev_right:
            self._vy -= self._vy_step
        if (space and not self._prev_space) or (r and not self._prev_r):
            self._vx = 0.0
            self._vy = 0.0
            self._omega = 0.0

        # Clamp
        self._vx = float(np.clip(self._vx, -self._max_v, self._max_v))
        self._vy = float(np.clip(self._vy, -self._max_v, self._max_v))

        self._prev_up = up
        self._prev_down = down
        self._prev_left = left
        self._prev_right = right
        self._prev_space = space
        self._prev_r = r

        return self.get_command()

    def get_command(self) -> np.ndarray:
        """Return current velocity command without updating keyboard state."""
        return np.array([self._vx, self._vy, self._omega], dtype=np.float32)

    def reset(self):
        """Reset command to zero."""
        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0

    def __repr__(self):
        return (
            f"KeyboardCommand(vx={self._vx:+.1f}, vy={self._vy:+.1f}, "
            f"omega={self._omega:+.1f})"
        )
