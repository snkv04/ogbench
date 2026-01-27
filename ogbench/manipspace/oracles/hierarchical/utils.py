import numpy as np
import cv2
from typing import Optional
import time
import gymnasium as gym
import pathlib
import imageio
from loguru import logger as logging
from typing import List


def init_realtime_rendering(window_name: str, width: int = 2000, height: int = 2000):
    """Initialize OpenCV window for real-time rendering.
    
    Args:
        window_name: Name of the OpenCV window.
        width: Window width in pixels.
        height: Window height in pixels.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)


def cleanup_realtime_rendering():
    """Cleanup OpenCV windows."""
    cv2.destroyAllWindows()


def add_text_overlay(
    frame: np.ndarray,
    option_idx: Optional[int] = None,
    option_text: Optional[str] = None,
    font_scale: float = 0.5,
    thickness: int = 2,
) -> np.ndarray:
    """Add HRL option info as text overlay on frame.
    
    Args:
        frame: RGB frame (numpy array).
        option_idx: Option index to display.
        option_text: Option name/description to display.
        font_scale: Font scale for text.
        thickness: Thickness of text.
    
    Returns:
        Frame with text overlay (copy of original).
    """
    if option_idx is None and option_text is None:
        return frame
    
    frame = frame.copy()  # Don't modify original
    
    # Build text string
    if option_idx is not None and option_text is not None:
        text = f"Option {option_idx}: {option_text}"
    elif option_idx is not None:
        text = f"Option {option_idx}"
    else:
        text = option_text
    
    # Draw text with black outline for visibility
    position = (10, 30)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (0, 0, 0), thickness + 2)  # Black outline
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness)  # White text
    
    return frame


def render_frame_realtime(
    env,
    window_name: str,
    delay: float,
    option_idx: Optional[int] = None,
    option_text: Optional[str] = None,
):
    """Render a frame in real-time using OpenCV.
    
    Args:
        env: The gymnasium environment.
        window_name: Name of the OpenCV window.
        delay: Time to sleep after rendering (seconds).
        option_idx: Optional option index to display as overlay.
        option_text: Optional option name to display as overlay.
    """
    frame = env.render()
    frame = add_text_overlay(frame, option_idx, option_text)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, frame_bgr)
    cv2.waitKey(1)
    time.sleep(delay)


def make_cube_env(
    env_id: str,
    seed: int,
    max_episode_steps: int,
    task_id: int | None,
    noise_initial_state: bool = True,
    reward_is_neg_dist: bool = False,
):
    env = gym.make(
        env_id,
        mode='task',
        terminate_at_goal=False,
        max_episode_steps=max_episode_steps,
        reward_task_id=task_id,  # Fixed task for all episodes (0 = default task, None = random)
        noise_initial_state=noise_initial_state,
        reward_is_neg_dist=reward_is_neg_dist,
    )
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def save_episode_video(
    frames: List[np.ndarray],
    save_dir: str,
    filename: str,
    fps: int = 30,
) -> str:
    """Save episode frames as a video file.
    
    Args:
        frames: List of RGB frames (numpy arrays).
        save_dir: Directory to save the video in.
        filename: Name of the video file (without extension).
        fps: Frames per second.
    
    Returns:
        Full path to the saved video.
    """
    if not frames:
        return ""
    
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    video_path = save_path / f"{filename}.mp4"
    
    with imageio.get_writer(
        video_path.as_posix(),
        fps=fps,
        codec='libx264',
        quality=8,
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
    
    logging.info(f"Saved video to: {video_path.as_posix()}")
    return video_path.as_posix()
