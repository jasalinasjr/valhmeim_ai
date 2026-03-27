import os
import time
import signal
import logging
import torch
import cv2
import numpy as np
import mss
import pydirectinput
import win32gui
import win32con
from pynvml import *

from gymnasium import spaces
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────
VALHEIM_WINDOW_TITLE = "Valheim"                    # Exact or partial window title
MODEL_SAVE = "valheim_ppo"
YOLO_MODEL_PATH = "valheim_custom_v3.pt"           # Set to None to disable

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMP_THRESHOLD = 82
VRAM_RESERVE = 450
MAX_BURST_STEPS = 2000

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler("valheim_ai.log", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# NVML GPU monitoring
nvml_handle = None
try:
    nvmlInit()
    nvml_handle = nvmlDeviceGetHandleByIndex(0)
    logger.info("NVML initialized successfully")
except Exception as e:
    logger.warning(f"NVML initialization failed: {e}")


def get_gpu_status():
    if not nvml_handle:
        return 0, 0, 9999
    try:
        temp = nvmlDeviceGetTemperature(nvml_handle, NVML_TEMPERATURE_GPU)
        mem = nvmlDeviceGetMemoryInfo(nvml_handle)
        used = mem.used // (1024 * 1024)
        free = (mem.total - mem.used) // (1024 * 1024)
        return temp, used, free
    except:
        return 0, 0, 9999


def find_valheim_window():
    """Find Valheim window and return its rect (left, top, right, bottom)"""
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if VALHEIM_WINDOW_TITLE.lower() in title.lower():
                rect = win32gui.GetWindowRect(hwnd)
                windows.append((hwnd, rect))
        return True

    windows = []
    win32gui.EnumWindows(callback, windows)
    
    if windows:
        hwnd, rect = windows[0]  # Take first match
        left, top, right, bottom = rect
        # Remove window borders/title bar for cleaner capture (approximate)
        capture_rect = {
            "top": top + 40,      # Skip title bar
            "left": left + 8,
            "width": right - left - 16,
            "height": bottom - top - 48
        }
        logger.debug(f"Found Valheim window: {win32gui.GetWindowText(hwnd)} at {rect}")
        return capture_rect
    return None


# ─── VALHEIM ENVIRONMENT WITH DYNAMIC WINDOW CAPTURE ───────────────────────────
class ValheimSimpleEnv(gym.Env):
    def __init__(self, image_size=(84, 84), frame_skip=4):
        super().__init__()
        self.frame_skip = frame_skip
        self.image_size = image_size

        self.sct = mss.mss()
        self.capture_region = None
        self.last_obs = None

        # Action Space
        self.action_space = spaces.Discrete(8)  # forward, back, left, right, jump, attack, interact, idle

        # Observation Space
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(image_size[0], image_size[1], 3),
            dtype=np.uint8
        )

        # Optional YOLO
        self.yolo = None
        if YOLO_MODEL_PATH and os.path.exists(YOLO_MODEL_PATH):
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(YOLO_MODEL_PATH)
                logger.info(f"YOLO model loaded: {YOLO_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load YOLO: {e}")

        self.current_step = 0
        self.episode_reward = 0.0

    def _update_capture_region(self):
        """Dynamically find and update Valheim window region"""
        region = find_valheim_window()
        if region:
            self.capture_region = region
        elif not self.capture_region:
            # Fallback to primary monitor
            monitor = self.sct.monitors[1]
            self.capture_region = {
                "top": 40, "left": 0,
                "width": monitor["width"], "height": monitor["height"] - 80
            }
            logger.warning("Valheim window not found. Using fallback capture region.")

    def _capture_screen(self):
        if not self.capture_region:
            self._update_capture_region()
        screenshot = self.sct.grab(self.capture_region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return img

    def _preprocess(self, img):
        return cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)

    def _get_reward(self, detections=None):
        reward = -0.01
        if detections:
            for det in detections:
                name = det.lower()
                if any(k in name for k in ["tree", "wood", "berry", "log"]):
                    reward += 0.8
                elif any(k in name for k in ["greydwarf", "troll", "wolf", "enemy"]):
                    reward -= 2.0
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_reward = 0.0
        self._update_capture_region()

        pydirectinput.press('esc')
        time.sleep(0.3)

        obs = self._capture_screen()
        self.last_obs = obs
        return self._preprocess(obs), {"episode_reward": 0.0}

    def step(self, action):
        self.current_step += 1

        action_map = {
            0: ("w", 0.25), 1: ("s", 0.25), 2: ("a", 0.25), 3: ("d", 0.25),
            4: ("space", 0.15), 5: ("left", 0.3), 6: ("e", 0.2), 7: (None, 0.1)
        }

        key, dur = action_map.get(action, (None, 0.1))
        if key:
            if key == "left":
                pydirectinput.mouseDown(button="left")
                time.sleep(dur)
                pydirectinput.mouseUp(button="left")
            else:
                pydirectinput.keyDown(key)
                time.sleep(dur)
                pydirectinput.keyUp(key)

        obs = self._capture_screen()
        self.last_obs = obs
        processed = self._preprocess(obs)

        detections = None
        if self.yolo:
            try:
                results = self.yolo(obs, verbose=False)
                detections = [results[0].names[int(c)] for c in results[0].boxes.cls]
            except:
                pass

        reward = self._get_reward(detections)
        self.episode_reward += reward

        terminated = False
        truncated = self.current_step > 4000

        info = {
            "episode_reward": self.episode_reward,
            "step": self.current_step,
            "detections": detections
        }

        return processed, reward, terminated, truncated, info

    def render(self):
        if self.last_obs is not None:
            cv2.imshow("Valheim Capture", cv2.cvtColor(self.last_obs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self):
        if hasattr(self, "sct"):
            self.sct.close()
        cv2.destroyAllWindows()


# ─── MAIN TRAINING LOOP ─────────────────────────────────────────────────────────
def main():
    running = True

    def shutdown(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received. Cleaning up...")
        running = False
        if nvml_handle:
            try:
                nvmlShutdown()
            except:
                pass

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    total_timesteps = 0

    while running:
        temp, used_vram, free_vram = get_gpu_status()
        logger.info(f"GPU Status | Temp: {temp}°C | Used VRAM: {used_vram}MB | Free: {free_vram}MB")

        if temp > TEMP_THRESHOLD or free_vram < VRAM_RESERVE:
            logger.warning(f"GPU condition critical (Temp {temp}°C or Free VRAM {free_vram}MB). Cooling...")
            time.sleep(60)
            continue

        # Create environment with dynamic window capture
        env = ValheimSimpleEnv(image_size=(84, 84))
        vec_env = DummyVecEnv([lambda: env])

        try:
            if os.path.exists(f"{MODEL_SAVE}.zip"):
                logger.info(f"Loading existing model: {MODEL_SAVE}")
                model = PPO.load(MODEL_SAVE, env=vec_env, device=DEVICE)
            else:
                logger.info("Creating new PPO model (CnnPolicy)")
                model = PPO(
                    "CnnPolicy",
                    vec_env,
                    verbose=1,
                    device=DEVICE,
                    learning_rate=3e-4,
                    n_steps=1024,
                    batch_size=64,
                    n_epochs=10,
                    ent_coef=0.01,
                    tensorboard_log="./valheim_ppo_logs/"
                )

            logger.info(f"Starting training burst of {MAX_BURST_STEPS} steps")
            model.learn(total_timesteps=MAX_BURST_STEPS, progress_bar=True)
            total_timesteps += MAX_BURST_STEPS

            logger.info(f"Saving model after {total_timesteps:,} total steps")
            model.save(MODEL_SAVE)

        except Exception as e:
            logger.error(f"Error during training: {e}")
        finally:
            # Cleanup
            vec_env.close()
            env.close()
            del model if 'model' in locals() else None
            del env, vec_env

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("Training loop terminated")
    if nvml_handle:
        nvmlShutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        if nvml_handle:
            try:
                nvmlShutdown()
            except:
                pass
