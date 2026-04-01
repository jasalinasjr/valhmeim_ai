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
import yaml

from gymnasium import spaces
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ========================= CONFIG =========================
CONFIG_PATH = "config.yaml"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: Config file '{CONFIG_PATH}' not found!")
        exit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required = ["VALHEIM_WINDOW_TITLE", "MODEL_SAVE", "YOLO_MODEL_PATH", "DEVICE",
                "TEMP_THRESHOLD", "VRAM_RESERVE", "MAX_BURST_STEPS", "MOUSE_SENSITIVITY",
                "REWARD_WEIGHTS", "capture_region", "action_map", "items", "enemies"]
    for key in required:
        if key not in config:
            print(f"ERROR: Missing required key '{key}' in config.yaml")
            exit(1)
    return config

config = load_config()

VALHEIM_WINDOW_TITLE = config["VALHEIM_WINDOW_TITLE"]
MODEL_SAVE = config["MODEL_SAVE"]
YOLO_MODEL_PATH = config["YOLO_MODEL_PATH"]
DEVICE = config["DEVICE"]
TEMP_THRESHOLD = config["TEMP_THRESHOLD"]
VRAM_RESERVE = config["VRAM_RESERVE"]
MAX_BURST_STEPS = config["MAX_BURST_STEPS"]
MOUSE_SENSITIVITY = config["MOUSE_SENSITIVITY"]
REWARD_WEIGHTS = config["REWARD_WEIGHTS"]
CAPTURE_REGION = config["capture_region"]
ACTION_MAP_RAW = config["action_map"]
ITEMS = config["items"]
ENEMIES = config["enemies"]

# Convert action_map
ACTION_MAP = {}
for k, v in ACTION_MAP_RAW.items():
    if isinstance(v, list) and len(v) == 2 and v[1] == "MOUSE_SENSITIVITY":
        ACTION_MAP[int(k)] = (v[0], MOUSE_SENSITIVITY)
    else:
        ACTION_MAP[int(k)] = tuple(v) if isinstance(v, list) else v

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler("valheim_ai.log", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

os.makedirs("debug_screenshots", exist_ok=True)

# GPU monitoring
nvml_handle = None
try:
    nvmlInit()
    nvml_handle = nvmlDeviceGetHandleByIndex(0)
except Exception as e:
    logger.warning(f"NVML init failed: {e}")

def get_gpu_status():
    if not nvml_handle:
        return 0, 0, 9999
    try:
        temp = nvmlDeviceGetTemperature(nvml_handle, NVML_TEMPERATURE_GPU)
        mem = nvmlDeviceGetMemoryInfo(nvml_handle)
        return temp, mem.used // (1024*1024), (mem.total - mem.used) // (1024*1024)
    except:
        return 0, 0, 9999

def find_valheim_window():
    def callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            if VALHEIM_WINDOW_TITLE.lower() in win32gui.GetWindowText(hwnd).lower():
                rect = win32gui.GetWindowRect(hwnd)
                windows.append((hwnd, rect))
        return True
    windows = []
    win32gui.EnumWindows(callback, windows)
    if windows:
        _, rect = windows[0]
        l, t, r, b = rect
        return {"top": t + 40, "left": l + 8, "width": r - l - 16, "height": b - t - 48}
    return None

class ValheimSimpleEnv(gym.Env):
    def __init__(self, image_size=(84, 84)):
        super().__init__()
        self.image_size = image_size
        self.detection_size = (640, 640)

        self.sct = mss.mss()
        self.capture_region = CAPTURE_REGION.copy()
        self.last_obs = None
        self.last_preprocessed = None
        self.last_detections = Counter()
        self.last_health_proxy = 0.5
        self.last_potential = 0.0

        self.action_space = spaces.Discrete(len(ACTION_MAP))
        self.observation_space = spaces.Box(0, 255, (image_size[0], image_size[1], 3), dtype=np.uint8)

        # YOLO11n Loading
        self.yolo = None
        if YOLO_MODEL_PATH and os.path.exists(YOLO_MODEL_PATH):
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(YOLO_MODEL_PATH)
                self.yolo.conf = 0.25
                self.yolo.iou = 0.45
                logger.info(f"✅ YOLO11n model loaded: {YOLO_MODEL_PATH} (conf={self.yolo.conf})")
                print("\nYOLO CLASSES:")
                for idx, name in self.yolo.names.items():
                    print(f"  {idx:2d}: {name}")
            except Exception as e:
                logger.error(f"YOLO load failed: {e}")
                self.yolo = None

        self.current_step = 0
        self.episode_reward = 0.0

    def _update_capture_region(self):
        region = find_valheim_window()
        if region:
            self.capture_region = region
        elif not self.capture_region:
            self.capture_region = CAPTURE_REGION.copy()

    def _capture_screen(self):
        self._update_capture_region()
        screenshot = self.sct.grab(self.capture_region)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    def _preprocess(self, img):
        return cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)

    def _detect_image(self, img):
        return cv2.resize(img, self.detection_size, interpolation=cv2.INTER_AREA)

    def _health_proxy(self, img: np.ndarray) -> float:
        try:
            h, w = img.shape[:2]
            region = img[max(0, h-130):max(0, h-40), 15:min(w, 170)]
            if region.size == 0:
                return 0.5
            hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([20, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 70, 70]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(mask1, mask2)
            red_ratio = np.sum(red_mask > 0) / red_mask.size
            return float(np.clip(red_ratio, 0.05, 1.0))
        except:
            return 0.5

    def _get_detections(self, img):
        if not self.yolo:
            return Counter()
        try:
            detect_img = self._detect_image(img)
            results = self.yolo(detect_img, verbose=False, conf=self.yolo.conf, iou=self.yolo.iou)[0]
            names = [results.names[int(c)] for c in results.boxes.cls]
            if self.current_step % 30 == 0:
                raw = {name: round(float(conf), 3) for name, conf in zip(names, results.boxes.conf.tolist())}
                logger.info(f"RAW YOLO | {raw}")
            return Counter(names)
        except Exception as e:
            if self.current_step % 30 == 0:
                logger.warning(f"YOLO inference failed: {e}")
            return Counter()

    def _compute_reward(self, current_detections: Counter, current_health: float,
                       current_preprocessed: np.ndarray, action: int) -> float:
        """Optimized reward with potential-based shaping."""
        reward = REWARD_WEIGHTS.get("time_penalty", -0.01)

        # Resource rewards
        resource_reward = 0.0
        for item in ITEMS:
            delta = current_detections.get(item, 0) - self.last_detections.get(item, 0)
            if delta > 0:
                bonus = REWARD_WEIGHTS.get(f"{item}_bonus", 
                                         3.0 if item in ["wood", "stone"] else 1.8)
                resource_reward += bonus * delta
        reward += resource_reward

        # Kill rewards
        kill_reward = 0.0
        for enemy in ENEMIES:
            delta = self.last_detections.get(enemy, 0) - current_detections.get(enemy, 0)
            if delta > 0:
                kill_reward += REWARD_WEIGHTS.get("kill_bonus", 10.0) * delta
        reward += kill_reward

        # Health rewards
        health_delta = current_health - self.last_health_proxy
        if health_delta > 0.06:
            reward += REWARD_WEIGHTS.get("health_gain_bonus", 5.0)
        elif health_delta < -0.06:
            reward += REWARD_WEIGHTS.get("health_loss_penalty", -4.0)

        # Enemy visible penalty
        enemy_visible = sum(1 for e in ENEMIES if e in current_detections)
        reward += enemy_visible * REWARD_WEIGHTS.get("enemy_visible_penalty", -1.2)

        # Look-down bonus
        if action == 23:
            reward += 0.8

        # Potential-based shaping
        current_potential = (current_health * 8.0) + len([k for k in current_detections if k in ITEMS]) * 2.5
        shaping = 0.99 * current_potential - getattr(self, 'last_potential', current_potential)
        reward += shaping
        self.last_potential = current_potential

        # Curiosity
        curiosity_reward = 0.0
        if self.last_preprocessed is not None:
            diff = np.abs(current_preprocessed.astype(np.float32) - self.last_preprocessed.astype(np.float32))
            curiosity = np.mean(diff) / 255.0
            curiosity_reward = REWARD_WEIGHTS.get("curiosity_scale", 0.35) * curiosity
            reward += curiosity_reward

        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_reward = 0.0
        self.last_detections = Counter()
        self.last_health_proxy = 0.5
        self.last_potential = 0.0

        self._update_capture_region()
        pydirectinput.press('esc')
        time.sleep(0.3)

        obs = self._capture_screen()
        self.last_obs = obs
        processed = self._preprocess(obs)
        self.last_preprocessed = processed.copy()

        return processed, {"episode_reward": 0.0}

    def step(self, action):
        self.current_step += 1

        # Action execution
        act = ACTION_MAP.get(action, (None, 0.1))
        action_name = "IDLE"
        if act[0] is not None:
            if act[0] == "left":
                action_name = "ATTACK"
                pydirectinput.mouseDown(button="left")
                time.sleep(act[1])
                pydirectinput.mouseUp(button="left")
            elif act[0] == "right":
                action_name = "BLOCK"
                pydirectinput.mouseDown(button="right")
                time.sleep(act[1])
                pydirectinput.mouseUp(button="right")
            elif act[0].startswith("mouse_"):
                direction = act[0]
                amount = act[1]
                if direction == "mouse_left":   pydirectinput.moveRel(xOffset=-amount, yOffset=0)
                elif direction == "mouse_right": pydirectinput.moveRel(xOffset=amount, yOffset=0)
                elif direction == "mouse_up":    pydirectinput.moveRel(xOffset=0, yOffset=-amount)
                elif direction == "mouse_down":  pydirectinput.moveRel(xOffset=0, yOffset=amount)
                action_name = direction.upper()
            else:
                action_name = act[0].upper()
                pydirectinput.keyDown(act[0])
                time.sleep(act[1])
                pydirectinput.keyUp(act[0])

        if self.current_step % 50 == 0:
            logger.info(f"Step {self.current_step:4d} | Action: {action} → {action_name}")

        # Capture frame
        obs = self._capture_screen()
        self.last_obs = obs
        processed = self._preprocess(obs)
        current_preprocessed = processed.copy()

        current_detections = self._get_detections(obs)
        current_health = self._health_proxy(obs)

        reward = self._compute_reward(current_detections, current_health, current_preprocessed, action)
        self.episode_reward += reward

        # Regular logging
        if self.current_step % 30 == 0:
            resources = {k: v for k, v in current_detections.items() if k in ITEMS}
            enemies_d = {k: v for k, v in current_detections.items() if k in ENEMIES}
            logger.info(f"Detections | Resources: {resources} | Enemies: {enemies_d} | Health: {current_health:.2f}")

        # Reward breakdown every 100 steps
        if self.current_step % 100 == 0:
            resource_r = sum(
                (REWARD_WEIGHTS.get(f"{i}_bonus", 3.0 if i in ["wood", "stone"] else 1.8)) *
                max(0, current_detections.get(i, 0) - self.last_detections.get(i, 0))
                for i in ITEMS
            )
            kill_r = sum(
                REWARD_WEIGHTS.get("kill_bonus", 10.0) *
                max(0, self.last_detections.get(e, 0) - current_detections.get(e, 0))
                for e in ENEMIES
            )
            health_delta = current_health - self.last_health_proxy
            health_r = (REWARD_WEIGHTS.get("health_gain_bonus", 5.0) if health_delta > 0.06 else
                       REWARD_WEIGHTS.get("health_loss_penalty", -4.0) if health_delta < -0.06 else 0.0)

            curiosity_r = 0.0
            if self.last_preprocessed is not None:
                diff = np.abs(current_preprocessed.astype(np.float32) - self.last_preprocessed.astype(np.float32))
                curiosity_r = REWARD_WEIGHTS.get("curiosity_scale", 0.35) * (np.mean(diff) / 255.0)

            logger.info(f"Reward Breakdown @ {self.current_step}: Time={REWARD_WEIGHTS.get('time_penalty',-0.01):.2f} | "
                        f"Resources={resource_r:.2f} | Kills={kill_r:.2f} | Health={health_r:.2f} | "
                        f"EnemyVis={-1.2 * sum(1 for e in ENEMIES if e in current_detections):.2f} | "
                        f"Curiosity={curiosity_r:.2f} | Total={reward:.2f}")

            # Debug screenshot
            try:
                if self.yolo:
                    debug_img = self._detect_image(self.last_obs.copy())
                    annotated = self.yolo(debug_img, verbose=False)[0].plot()
                    filename = f"debug_screenshots/step_{self.current_step:06d}.jpg"
                    cv2.imwrite(filename, annotated)
                    logger.info(f"Debug screenshot saved: {filename}")
            except Exception as e:
                logger.warning(f"Debug screenshot failed: {e}")

        # Update last states safely
        self.last_detections = current_detections.copy()
        self.last_health_proxy = current_health
        self.last_preprocessed = current_preprocessed

        terminated = False
        truncated = self.current_step > 8000   # Increased for better training

        return processed, reward, terminated, truncated, {"episode_reward": self.episode_reward}

    def close(self):
        if hasattr(self, "sct"):
            self.sct.close()
        cv2.destroyAllWindows()

# ========================= MAIN =========================
def main():
    running = True
    def shutdown(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received.")
        running = False
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    total_timesteps = 0

    while running:
        temp, used, free = get_gpu_status()
        logger.info(f"GPU | Temp: {temp}°C | Used: {used}MB | Free: {free}MB")

        if temp > TEMP_THRESHOLD or free < VRAM_RESERVE:
            logger.warning("GPU cooling pause (60s)...")
            time.sleep(60)
            continue

        env = ValheimSimpleEnv()
        vec_env = DummyVecEnv([lambda: env])

        try:
            model_path = f"{MODEL_SAVE}.zip"
            if os.path.exists(model_path):
                logger.info(f"Loading existing model: {MODEL_SAVE}")
                model = PPO.load(model_path, env=vec_env, device=DEVICE)
            else:
                logger.info("Creating new PPO model")
                model = PPO("CnnPolicy", vec_env, device=DEVICE,
                            learning_rate=3e-4, n_steps=1024, batch_size=64,
                            n_epochs=10, ent_coef=0.01, tensorboard_log="./valheim_ppo_logs/")

            logger.info(f"Starting burst of {MAX_BURST_STEPS} steps")
            model.learn(total_timesteps=MAX_BURST_STEPS, progress_bar=True)
            total_timesteps += MAX_BURST_STEPS
            model.save(MODEL_SAVE)
            logger.info(f"Model saved after {total_timesteps:,} total steps")

        except Exception as e:
            logger.error(f"Training error: {e}")
            if "spaces do not match" in str(e).lower() and os.path.exists(f"{MODEL_SAVE}.zip"):
                logger.info("Space mismatch → deleting old model")
                os.remove(f"{MODEL_SAVE}.zip")
        finally:
            vec_env.close()
            env.close()
            if 'model' in locals():
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if nvml_handle:
        try:
            nvmlShutdown()
        except:
            pass

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
