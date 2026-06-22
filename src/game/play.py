"""
Bla bla bla
"""
import yaml
import cv2
import torch

import argparse
import pygame

from src.game.snake import SnakeEnv
from src.policy.policy import Policy, PolicyDQN, PolicyPPO
from src.utils.utils import *


class PlayEnv:
    """Environment for playing the game with a human or a bot."""
    def __init__(self, modality, config, record):
        self.mode = modality
        self.config = config
        self.record = record

        self.env = SnakeEnv(**config)
        self.policy = PolicyDQN(**config)

        if self.record:
            self.video = cv2.VideoWriter(
                f"video/game_difficulty_{config.get("difficulty")}.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                config.get("fps"),
                (config.get("width"), config.get("height"))
            )




    def _play_human(self):
        obs, info = self.env.reset()

        if self.record:
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            self.video.write(frame)

        done = False
        trunc = False
        current_action = 0
        while not done and not trunc:
            if self.env.window is not None:
                for event in pygame.event.get(pygame.KEYDOWN):
                    if event.key == pygame.K_w: current_action = 0
                    elif event.key == pygame.K_s: current_action = 1
                    elif event.key == pygame.K_a: current_action = 2
                    elif event.key == pygame.K_d: current_action = 3

            obs, rew, done, trunc, info = self.env.step(current_action)

            if self.record:
                frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                self.video.write(frame)

            if not self.env.render():
                break

        self.env.close()


    def _play_bot(self):
        obs, info = self.env.reset()

        if self.record:
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            self.video.write(frame)

        done = False
        trunc = False

        obs = self.policy._format_state(obs, bracket= True)
        action = self.policy.get_action(obs, greedy = True)[0]
        curr_frame = 0

        while not done and not trunc:
            next_frame = curr_frame + 1
            if next_frame % 3 == 0:
                print("I am not here", curr_frame)
                obs = self.policy._format_state(obs, bracket= True)
                new_action, info = self.policy.get_action(obs, greedy = True)
                print(info)
                obs, rew, done, trunc, info = self.env.step(new_action)
            else:
                print("I am here", curr_frame)
                obs, rew, done, trunc, info = self.env.step(action)

            if self.record:
                frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                self.video.write(frame)

            curr_frame += 1

            if not self.env.render():
                break

        self.env.close()


    def play(self):
        if self.mode == "human":
            self._play_human()
        else:
            self._play_bot()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helper for PlayEnv")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--modality", type=str, default="human", choices=["human", "bot"], help="Modality of the environment.")
    parser.add_argument("--record", type=bool, default=False, help="Whether to record the gameplay.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = flat_config(config)

    env = PlayEnv(args.modality, config, args.record)
    env.play()


