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
from collections import Counter
from pynvml import *

from gymnasium import spaces
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────
VALHEIM_WINDOW_TITLE = "Valheim"
MODEL_SAVE = "valheim_ppo"
YOLO_MODEL_PATH = "valheim_custom_v3.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMP_THRESHOLD = 89
VRAM_RESERVE = 450
MAX_BURST_STEPS = 2000

MOUSE_SENSITIVITY = 28

REWARD_WEIGHTS = {
    "time_penalty": -0.01,
    "wood_bonus": 2.0,
    "kill_bonus": 5.0,
    "health_bonus": 3.0,
    "curiosity_scale": 0.5,
    "logout_penalty": -50.0,
    "empty_inventory_penalty": -0.8
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler("valheim_ai.log", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
        _, rect = windows[0]
        left, top, right, bottom = rect
        return {
            "top": top + 40,
            "left": left + 8,
            "width": right - left - 16,
            "height": bottom - top - 48
        }
    return None


# ─── VALHEIM ENVIRONMENT ────────────────────────────────────────────────────────
class ValheimSimpleEnv(gym.Env):
    def __init__(self, image_size=(84, 84)):
        super().__init__()
        self.image_size = image_size

        self.sct = mss.mss()
        self.capture_region = None
        self.last_obs = None
        self.last_preprocessed = None
        self.last_detections = Counter()
        self.last_health_proxy = 0.0
        self.last_red_pixel_ratio = 0.0

        # 16 actions: 0-7 movement, 8-11 mouse look
        self.action_space = spaces.Discrete(16)

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(image_size[0], image_size[1], 3),
            dtype=np.uint8
        )

        self.yolo = None
        if YOLO_MODEL_PATH and os.path.exists(YOLO_MODEL_PATH):
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(YOLO_MODEL_PATH)
                logger.info(f"YOLO loaded: {YOLO_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"YOLO load failed: {e}")

        self.current_step = 0
        self.episode_reward = 0.0

    def _update_capture_region(self):
        region = find_valheim_window()
        if region:
            self.capture_region = region
        elif not self.capture_region:
            monitor = self.sct.monitors[1]
            self.capture_region = {"top": 40, "left": 0, "width": monitor["width"], "height": monitor["height"] - 80}

    def _capture_screen(self):
        if not self.capture_region:
            self._update_capture_region()
        screenshot = self.sct.grab(self.capture_region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return img

    def _preprocess(self, img):
        return cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)

    def _health_proxy(self, img: np.ndarray) -> float:
        """Health proxy targeting area 6 (bottom-left red health bar)"""
        h, w = img.shape[:2]
        # Refined coordinates based on screenshot (area 6)
        y_start = max(0, h - 140)
        y_end   = max(0, h - 35)
        x_start = 20
        x_end   = min(w, 180)
        
        health_region = img[y_start:y_end, x_start:x_end]
        if health_region.size == 0:
            return 0.5
        
        hsv = cv2.cvtColor(health_region, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 80, 80]),   np.array([15, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([165, 80, 80]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        red_ratio = np.sum(red_mask > 0) / red_mask.size
        
        if np.mean(health_region) < 40:
            return max(0.05, red_ratio * 0.6)
        
        return float(np.clip(red_ratio, 0.05, 1.0))

    def _is_logout_or_quit(self, current_red_ratio: float) -> bool:
        if current_red_ratio < 0.08 and self.last_red_pixel_ratio > 0.25:
            return True
        return False

    def _get_detections(self, img):
        if not self.yolo:
            return Counter()
        try:
            results = self.yolo(img, verbose=False)[0]
            names = [results.names[int(c)] for c in results.boxes.cls]
            return Counter(names)
        except:
            return Counter()

    def _is_inventory_empty(self, detections: Counter) -> bool:
        """Improved using hotbar (area 1)"""
        common_items = {"wood", "stone", "axe", "hammer", "berry", "meat", "log", "resin", "torch", "pickaxe"}
        detected = {k.lower() for k in detections.keys()}
        return len(common_items & detected) == 0

    def _compute_reward(self, current_detections: Counter, current_health: float, current_red_ratio: float) -> float:
        reward = REWARD_WEIGHTS["time_penalty"]

        # Resource collection (hotbar / ground items)
        for item in ["wood", "berry", "log", "pinecone", "resin"]:
            delta = current_detections[item] - self.last_detections[item]
            if delta > 0:
                reward += REWARD_WEIGHTS["wood_bonus"] * delta

        # Kill bonuses
        for enemy in ["greyling", "boar", "greydwarf", "troll", "wolf", "skeleton", "enemy"]:
            delta = self.last_detections[enemy] - current_detections[enemy]
            if delta > 0:
                reward += REWARD_WEIGHTS["kill_bonus"] * delta

        # Health change
        health_delta = current_health - self.last_health_proxy
        if health_delta > 0.08:
            reward += REWARD_WEIGHTS["health_bonus"] * 1.5
        elif health_delta < -0.08:
            reward -= 1.5

        # Logout / Quit penalty
        if self._is_logout_or_quit(current_red_ratio):
            reward += REWARD_WEIGHTS["logout_penalty"]
            logger.warning("Logout/Quit detected! Large penalty applied.")

        # Empty inventory penalty (checks hotbar area 1)
        if self._is_inventory_empty(current_detections):
            reward += REWARD_WEIGHTS["empty_inventory_penalty"]

        # Curiosity
        if self.last_preprocessed is not None:
            diff = np.abs(current_preprocessed.astype(np.float32) - self.last_preprocessed.astype(np.float32))
            curiosity = np.mean(diff) / 255.0
            reward += REWARD_WEIGHTS["curiosity_scale"] * curiosity

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_reward = 0.0
        self.last_detections = Counter()
        self.last_health_proxy = 0.0
        self.last_red_pixel_ratio = 0.0
        self._update_capture_region()

        pydirectinput.press('esc')
        time.sleep(0.3)

        obs = self._capture_screen()
        self.last_obs = obs
        processed = self._preprocess(obs)
        self.last_preprocessed = processed.copy()

        time.sleep(0.3)
        pydirectinput.press('esc')

        return processed, {"episode_reward": 0.0}

    def step(self, action):
        self.current_step += 1

        # Action Map (16 actions)
        action_map = {
            0: ("w", 0.25), 1: ("s", 0.25), 2: ("a", 0.25), 3: ("d", 0.25),
            4: ("space", 0.15), 5: ("left", 0.3), 6: ("e", 0.2), 7: (None, 0.1),
            # Mouse look - critical for looking down at ground resources (area 9)
            8:  ("mouse_left",  MOUSE_SENSITIVITY),
            9:  ("mouse_right", MOUSE_SENSITIVITY),
            10: ("mouse_up",    MOUSE_SENSITIVITY),
            11: ("mouse_down",  MOUSE_SENSITIVITY),
        }

        act = action_map.get(action, (None, 0.1))
        if act[0] is not None:
            if act[0] == "left":
                pydirectinput.mouseDown(button="left")
                time.sleep(act[1])
                pydirectinput.mouseUp(button="left")
            elif act[0].startswith("mouse_"):
                direction = act[0]
                amount = act[1]
                if direction == "mouse_left":
                    pydirectinput.moveRel(xOffset=-amount, yOffset=0)
                elif direction == "mouse_right":
                    pydirectinput.moveRel(xOffset=amount, yOffset=0)
                elif direction == "mouse_up":
                    pydirectinput.moveRel(xOffset=0, yOffset=-amount)
                elif direction == "mouse_down":
                    pydirectinput.moveRel(xOffset=0, yOffset=amount)
            else:
                pydirectinput.keyDown(act[0])
                time.sleep(act[1])
                pydirectinput.keyUp(act[0])

        obs = self._capture_screen()
        self.last_obs = obs
        processed = self._preprocess(obs)
        current_preprocessed = processed.copy()

        current_detections = self._get_detections(obs)
        current_health = self._health_proxy(obs)
        current_red_ratio = current_health

        reward = self._compute_reward(current_detections, current_health, current_red_ratio)
        self.episode_reward += reward

        self.last_detections = current_detections
        self.last_health_proxy = current_health
        self.last_red_pixel_ratio = current_red_ratio
        self.last_preprocessed = current_preprocessed

        terminated = False
        truncated = self.current_step > 4000

        info = {
            "episode_reward": self.episode_reward,
            "step": self.current_step,
            "detections": dict(current_detections),
            "health_proxy": current_health
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
        logger.info(f"GPU | Temp: {temp}°C | Used: {used_vram}MB | Free: {free_vram}MB")

        if temp > TEMP_THRESHOLD or free_vram < VRAM_RESERVE:
            logger.warning("GPU cooling pause...")
            time.sleep(60)
            continue

        env = ValheimSimpleEnv(image_size=(84, 84))
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecFrameStack(vec_env, n_stack=4)

        try:
            if os.path.exists(f"{MODEL_SAVE}.zip"):
                logger.info(f"Loading model {MODEL_SAVE}")
                model = PPO.load(MODEL_SAVE, env=vec_env, device=DEVICE)
            else:
                logger.info("Creating new PPO model")
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

            logger.info(f"Training burst — {MAX_BURST_STEPS} steps")
            model.learn(total_timesteps=MAX_BURST_STEPS, progress_bar=True)
            total_timesteps += MAX_BURST_STEPS

            logger.info(f"Saving model after {total_timesteps:,} total steps")
            model.save(MODEL_SAVE)

        except Exception as e:
            logger.error(f"Training error: {e}")
        finally:
            vec_env.close()
            env.close()
            if 'model' in locals():
                del model
            del env, vec_env

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("Training loop ended")
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
