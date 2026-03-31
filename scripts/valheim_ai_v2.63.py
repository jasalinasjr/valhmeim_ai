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

# ─── CONFIGURATION LOADING ─────────────────────────────────────────────────────
CONFIG_PATH = "config.yaml"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: Config file '{CONFIG_PATH}' not found!")
        exit(1)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        exit(1)

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
PPO_PARAMS = config.get("PPO_PARAMS", {
    "policy": "CnnPolicy",
    "verbose": 1,
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "ent_coef": 0.01,
    "tensorboard_log": "./valheim_ppo_logs/"
})

# Convert action_map
ACTION_MAP = {}
for key, value in ACTION_MAP_RAW.items():
    if isinstance(value, list):
        if len(value) >= 2 and value[1] == "MOUSE_SENSITIVITY":
            ACTION_MAP[int(key)] = (f"mouse_{value[0].split('_')[-1]}", MOUSE_SENSITIVITY)
        else:
            ACTION_MAP[int(key)] = tuple(value)
    else:
        ACTION_MAP[int(key)] = value

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler("valheim_ai.log", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

os.makedirs("debug_screenshots", exist_ok=True)

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
        return {"top": top + 40, "left": left + 8, "width": right - left - 16, "height": bottom - top - 48}
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
        self.observation_space = spaces.Box(low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8)

        self.yolo = None
        if YOLO_MODEL_PATH and os.path.exists(YOLO_MODEL_PATH):
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(YOLO_MODEL_PATH)
                self.yolo.conf = 0.20   # Lowered for more detections
                logger.info(f"YOLO loaded successfully: {YOLO_MODEL_PATH} (conf=0.20)")
                print("\nYOLO CLASSES:")
                for idx, name in self.yolo.names.items():
                    print(f"  {idx:2d}: {name}")
            except Exception as e:
                logger.error(f"YOLO load failed: {e}")
                print(f"\nERROR: Could not load YOLO model. Error: {e}")
                print("Make sure valheim_custom_v3.pt is a valid Ultralytics YOLO .pt file.")
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
        if not self.capture_region:
            self._update_capture_region()
        screenshot = self.sct.grab(self.capture_region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return img

    def _preprocess(self, img):
        return cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)

    def _detect_image(self, img):
        return cv2.resize(img, self.detection_size, interpolation=cv2.INTER_AREA)

    def _health_proxy(self, img: np.ndarray) -> float:
        try:
            h, w = img.shape[:2]
            y_start = max(0, h - 130)
            y_end   = max(0, h - 40)
            x_start = 15
            x_end   = min(w, 170)

            health_region = img[y_start:y_end, x_start:x_end]
            if health_region.size == 0:
                return 0.5

            hsv = cv2.cvtColor(health_region, cv2.COLOR_RGB2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([20, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 70, 70]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(mask1, mask2)

            red_ratio = np.sum(red_mask > 0) / red_mask.size
            mean_val = np.mean(health_region)

            if self.current_step % 30 == 0:
                logger.info(f"Health Proxy Debug | red_ratio={red_ratio:.3f} | mean={mean_val:.1f} | region_shape={health_region.shape}")

            if mean_val < 40:
                return max(0.05, red_ratio * 0.6)
            return float(np.clip(red_ratio, 0.05, 1.0))
        except Exception as e:
            if self.current_step % 30 == 0:
                logger.warning(f"Health proxy error: {e}")
            return 0.5

    def _get_detections(self, img):
        if not self.yolo:
            return Counter()
        try:
            detect_img = self._detect_image(img)
            results = self.yolo(detect_img, verbose=False)[0]
            names = [results.names[int(c)] for c in results.boxes.cls]
            confs = results.boxes.conf.tolist()

            if self.current_step % 30 == 0:
                raw = {name: round(conf, 2) for name, conf in zip(names, confs)}
                logger.info(f"RAW YOLO | {raw}")

            return Counter(names)
        except Exception as e:
            if self.current_step % 30 == 0:
                logger.warning(f"YOLO inference failed: {e}")
            return Counter()

    def _compute_reward(self, current_detections: Counter, current_health: float, 
                       current_preprocessed: np.ndarray, action: int) -> float:
        reward = REWARD_WEIGHTS["time_penalty"]

        resource_reward = sum(
            REWARD_WEIGHTS["wood_bonus"] * (current_detections.get(item, 0) - self.last_detections.get(item, 0))
            for item in ITEMS 
            if current_detections.get(item, 0) > self.last_detections.get(item, 0)
        )
        reward += resource_reward

        kill_reward = sum(
            REWARD_WEIGHTS["kill_bonus"] * (self.last_detections.get(enemy, 0) - current_detections.get(enemy, 0))
            for enemy in ENEMIES 
            if self.last_detections.get(enemy, 0) > current_detections.get(enemy, 0)
        )
        reward += kill_reward

        health_delta = current_health - self.last_health_proxy
        health_reward = (
            REWARD_WEIGHTS["health_gain_bonus"] if health_delta > 0.08 else
            (REWARD_WEIGHTS["health_loss_penalty"] if health_delta < -0.08 else 0.0)
        )
        reward += health_reward

        enemy_visible_reward = sum(
            REWARD_WEIGHTS["enemy_visible_penalty"] 
            for enemy in ENEMIES 
            if enemy in current_detections
        )
        reward += enemy_visible_reward

        if self.current_step > 0 and action == 23:
            reward += 0.6

        potential = current_health * 8.0 + len([k for k in current_detections if k in ITEMS]) * 3.0
        if hasattr(self, 'last_potential'):
            shaping = 0.99 * potential - self.last_potential
            reward += shaping
        self.last_potential = potential

        curiosity_reward = 0.0
        if self.last_preprocessed is not None:
            diff = np.abs(current_preprocessed.astype(np.float32) - self.last_preprocessed.astype(np.float32))
            curiosity = np.mean(diff) / 255.0
            curiosity_reward = REWARD_WEIGHTS["curiosity_scale"] * curiosity
            reward += curiosity_reward

        return reward

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

        act = ACTION_MAP.get(action, (None, 0.1))
        action_name = "unknown"

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
                if direction == "mouse_left":
                    action_name = "LOOK LEFT"
                    pydirectinput.moveRel(xOffset=-amount, yOffset=0)
                elif direction == "mouse_right":
                    action_name = "LOOK RIGHT"
                    pydirectinput.moveRel(xOffset=amount, yOffset=0)
                elif direction == "mouse_up":
                    action_name = "LOOK UP"
                    pydirectinput.moveRel(xOffset=0, yOffset=-amount)
                elif direction == "mouse_down":
                    action_name = "LOOK DOWN"
                    pydirectinput.moveRel(xOffset=0, yOffset=amount)
            else:
                action_name = f"{act[0].upper()}"
                pydirectinput.keyDown(act[0])
                time.sleep(act[1])
                pydirectinput.keyUp(act[0])
        else:
            action_name = "IDLE"

        if self.current_step % 50 == 0:
            logger.info(f"Step {self.current_step:4d} | Action: {action} → {action_name}")

        obs = self._capture_screen()
        self.last_obs = obs
        processed = self._preprocess(obs)
        current_preprocessed = processed.copy()

        current_detections = self._get_detections(obs)
        current_health = self._health_proxy(obs)

        reward = self._compute_reward(current_detections, current_health, current_preprocessed, action)
        self.episode_reward += reward

        if self.current_step % 30 == 0:
            resources = {k: v for k, v in current_detections.items() if k in ITEMS}
            enemies_detected = {k: v for k, v in current_detections.items() if k in ENEMIES}
            logger.info(f"Detections | Resources: {resources} | Enemies: {enemies_detected} | Health: {current_health:.2f}")

        if self.current_step % 100 == 0:
            resource_reward = sum(
                REWARD_WEIGHTS["wood_bonus"] * (current_detections.get(item, 0) - self.last_detections.get(item, 0))
                for item in ITEMS if current_detections.get(item, 0) > self.last_detections.get(item, 0)
            )
            kill_reward = sum(
                REWARD_WEIGHTS["kill_bonus"] * (self.last_detections.get(enemy, 0) - current_detections.get(enemy, 0))
                for enemy in ENEMIES if self.last_detections.get(enemy, 0) > current_detections.get(enemy, 0)
            )
            health_delta = current_health - self.last_health_proxy
            health_reward = (
                REWARD_WEIGHTS["health_gain_bonus"] if health_delta > 0.08 else
                (REWARD_WEIGHTS["health_loss_penalty"] if health_delta < -0.08 else 0.0)
            )
            enemy_visible_reward = sum(
                REWARD_WEIGHTS["enemy_visible_penalty"] for enemy in ENEMIES if enemy in current_detections
            )
            curiosity_reward = 0.0
            if self.last_preprocessed is not None:
                diff = np.abs(current_preprocessed.astype(np.float32) - self.last_preprocessed.astype(np.float32))
                curiosity_reward = REWARD_WEIGHTS["curiosity_scale"] * (np.mean(diff) / 255.0)

            logger.info(f"Reward Breakdown @ Step {self.current_step}: "
                        f"Time={REWARD_WEIGHTS['time_penalty']:.2f} | "
                        f"Resources={resource_reward:.2f} | "
                        f"Kills={kill_reward:.2f} | "
                        f"Health={health_reward:.2f} | "
                        f"EnemyVisible={enemy_visible_reward:.2f} | "
                        f"Curiosity={curiosity_reward:.2f} | "
                        f"Total={reward:.2f}")

            # Save debug screenshot with detections
            try:
                debug_img = self._detect_image(self.last_obs.copy())
                if self.yolo:
                    results = self.yolo(debug_img, verbose=False)[0]
                    annotated = results.plot()
                    filename = f"debug_screenshots/step_{self.current_step:06d}.jpg"
                    cv2.imwrite(filename, annotated)
                    logger.info(f"Debug screenshot saved: {filename}")
            except Exception as e:
                if self.current_step % 100 == 0:
                    logger.warning(f"Failed to save debug screenshot: {e}")

        self.last_detections = current_detections
        self.last_health_proxy = current_health
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

    if YOLO_MODEL_PATH and os.path.exists(YOLO_MODEL_PATH):
        try:
            from ultralytics import YOLO
            temp_model = YOLO(YOLO_MODEL_PATH)
            print("\n" + "="*60)
            print("YOLO MODEL DEBUG INFORMATION")
            print("="*60)
            print("Detected classes:")
            for idx, name in temp_model.names.items():
                print(f"  {idx:2d}: {name}")
            print(f"Total classes: {len(temp_model.names)}")
            print("="*60 + "\n")
            del temp_model
        except Exception as e:
            logger.error(f"Could not load YOLO for debug: {e}")
            print(f"\nERROR: Could not load YOLO model for debug. Error: {e}")

    while running:
        temp, used_vram, free_vram = get_gpu_status()
        logger.info(f"GPU | Temp: {temp}°C | Used: {used_vram}MB | Free: {free_vram}MB")

        if temp > TEMP_THRESHOLD or free_vram < VRAM_RESERVE:
            logger.warning("GPU cooling pause...")
            time.sleep(60)
            continue

        env = ValheimSimpleEnv(image_size=(84, 84))
        vec_env = DummyVecEnv([lambda: env])

        try:
            model_path = f"{MODEL_SAVE}.zip"
            if os.path.exists(model_path):
                logger.info(f"Loading existing model: {MODEL_SAVE}")
                model = PPO.load(model_path, env=vec_env, device=DEVICE)
            else:
                logger.info("No saved model found. Creating new PPO model.")
                model = PPO(
                    env=vec_env,
                    device=DEVICE,
                    **PPO_PARAMS
                )

            logger.info(f"Training burst — {MAX_BURST_STEPS} steps")
            model.learn(total_timesteps=MAX_BURST_STEPS, progress_bar=True)
            total_timesteps += MAX_BURST_STEPS

            logger.info(f"Saving model after {total_timesteps:,} total steps")
            model.save(MODEL_SAVE)

        except Exception as e:
            logger.error(f"Training error: {e}")
            if ("spaces do not match" in str(e).lower()) and os.path.exists(f"{MODEL_SAVE}.zip"):
                logger.info("Space mismatch detected. Deleting old model...")
                os.remove(f"{MODEL_SAVE}.zip")
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
