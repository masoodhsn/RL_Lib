from stable_baselines3 import PPO
from typing import Optional
import gymnasium as gym


class PPOAgent:
    """
    PPOAgent - Agent مخصوص الگوریتم PPO از stable-baselines3
    ----------------------------------------------------------
    کاربر باید خودش محیط (env) را بسازد و به این کلاس بدهد.
    """

    def __init__(self, env: gym.Env, policy: str = "MlpPolicy", **kwargs):
        if not isinstance(env, gym.Env):
            raise TypeError("❌ محیط باید از gym.Env ارث‌بری کرده باشد.")
        self.env = env
        self.policy = policy
        self.model = PPO(self.policy, self.env, verbose=1, **kwargs)

    def train(self, total_timesteps: int = 10000):
        """
        آموزش عامل برای تعداد مشخصی timestep
        """
        print(f"🎯 شروع آموزش برای {total_timesteps} گام...")
        self.model.learn(total_timesteps=total_timesteps)
        print("✅ آموزش به پایان رسید!")

    def save(self, path: str = "ppo_agent.zip"):
        """
        ذخیره مدل آموزش‌دیده
        """
        self.model.save(path)
        print(f"💾 مدل ذخیره شد در: {path}")

    def load(self, path: str):
        """
        بارگذاری مدل ذخیره‌شده
        """
        print(f"📂 در حال بارگذاری مدل از: {path}")
        self.model = PPO.load(path)
        print("✅ مدل با موفقیت بارگذاری شد!")

    def get_model(self):
        """
        برگرداندن شیء مدل PPO (در صورت نیاز کاربر)
        """
        return self.model
