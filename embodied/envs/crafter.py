import json

import embodied
import numpy as np
from ..run.run_utils import ImageUtil


class Crafter(embodied.Env):

  def __init__(self, task, size=(64, 64), logs=False, logdir=None, seed=None):
    assert task in ('reward', 'noreward')
    import crafter
    self._env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
    self._logs = logs
    self._logdir = logdir and embodied.Path(logdir)
    self._logdir and self._logdir.mkdir()
    self._episode = 0
    self._length = None
    self._reward = None
    self._achievements = crafter.constants.achievements.copy()
    self._done = True
    self.visualize = False
    self._health_map = None
    if self.visualize:
      self._image_util = ImageUtil(str(self._logdir), experiment_label='crafter')

  @property
  def obs_space(self):
    spaces = {
        'image': embodied.Space(np.uint8, self._env.observation_space.shape),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        'log_reward': embodied.Space(np.float32),
        'grayscale': embodied.Space(np.uint8, self._env.observation_space.shape),
        'semantic': embodied.Space(np.uint8, shape=(64, 64, 1)),
        'danger': embodied.Space(np.uint8, shape=(64, 64, 3)),
        'health': embodied.Space(np.uint8, shape=(64, 64, 1))
    }
    if self._logs:
      spaces.update({
          f'log_achievement_{k}': embodied.Space(np.int32)
          for k in self._achievements})
    return spaces

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._episode += 1
      self._length = 0
      self._reward = 0
      self._done = False
      self._image_count = 0
      
      image = self._env.reset()
      self._prev_inventory = self._env._player.inventory.copy()
      
      if self.visualize:
        self._save_image(image)
      
      self._health_map = np.zeros((64, 64), dtype=np.uint8)
      return self._obs(image, 0.0, {}, is_first=True)
    
    image, reward, self._done, info = self._env.step(action['action']) # gets default reward from Env step function
    
    
    ### Reward shaping - item collection task
    # current_inventory = info['inventory']
    # if self._prev_inventory:
    #   # Reward for any positive increase in inventory counts
    #   for item, count in current_inventory.items():
    #       prev_count = self._prev_inventory.get(item, 0)
    #       diff = count - prev_count
    #       if diff > 0:
    #           reward += 0.1 * diff  # scale as you like
    # self._prev_inventory = current_inventory.copy()
    # # Track reward and inventory count changes
    # print(f"[Step {self._length}] Reward: {reward:.2f}")
    # print("Inventory:")
    # for item, count in current_inventory.items():
    #     prev_count = self._prev_inventory.get(item, 0) if self._prev_inventory else 0
    #     if count != prev_count:
    #         print(f"  {item}: {prev_count} -> {count} (+{count - prev_count})")
    # print("-" * 30)
    
    self._reward = reward
    
    self._length += 1
    if self._done and self._logdir:
      self._write_stats(self._length, self._reward, info)
    
    # save the following images in a folder
    # ask GPT to write a script to animate the images for visualization
    if self.visualize:
      print("visualizing")
      self._save_image(image)
    
    return self._obs(
        image, reward, info,
        is_last=self._done,
        is_terminal=info['discount'] == 0)
  
  def _save_image(self, image):
    image_name = f"episode{self._episode:03d}_frame{self._image_count:05d}.png"
    self._image_util.print_image(
        image_mat=image,
        image_folder=self._image_util.actual_image_folder,
        image_name=image_name,
        is_normalized=False
    )
    self._image_count += 1

  def _obs(
      self, image, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    
    grayscale = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    grayscale = np.expand_dims(grayscale, axis=-1)
    grayscale = np.repeat(grayscale, 3, axis=-1)
    
    if "semantic" in info:
      semantic = info["semantic"]
    else:
      semantic = self._env._sem_view()
    semantic = np.expand_dims(semantic, axis=-1)
    semantic = (semantic / 12.0 * 255).astype(np.uint8)
    
    danger = self._danger_heatmap()
    
    player = self._env._player
    px, py = player.pos
    self._health_map[px, py] = player.health
    scaled_health_map = (self._health_map / 10 * 255).astype(np.uint8) # Max player health is 9
    health = np.expand_dims(scaled_health_map, axis=-1)
    
    obs = dict(
        image=image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        log_reward=np.float32(info['reward'] if info else 0.0),
        grayscale=grayscale,
        semantic=semantic,
        danger=danger,
        health=health
    )
    if self._logs:
      log_achievements = {
          f'log_achievement_{k}': info['achievements'][k] if info else 0
          for k in self._achievements}
      obs.update({k: np.int32(v) for k, v in log_achievements.items()})
    return obs

  def _write_stats(self, length, reward, info):
    stats = {
        'episode': self._episode,
        'length': length,
        'reward': round(reward, 1),
        **{f'achievement_{k}': v for k, v in info['achievements'].items()},
    }
    filename = self._logdir / 'stats.jsonl'
    lines = filename.read() if filename.exists() else ''
    lines += json.dumps(stats) + '\n'
    filename.write(lines)
    print(f'Wrote stats: {filename}')
  
  def _danger_heatmap(self):
    world = self._env._world
    H, W = world.area
    heatmap = np.zeros((H, W), dtype=np.float32)
    
    def apply_gaussian(heatmap, pos, intensity=1.0, sigma=3.0):
      for x in range(H):
        for y in range(W):
          dist = np.linalg.norm(np.array([x, y]) - np.array(pos))
          heatmap[x, y] += intensity * np.exp(-dist**2 / (2 * sigma**2))
    
    for obj in world.objects:
      if obj.__class__.__name__ in ['Zombie', 'Skeleton']:
        apply_gaussian(heatmap, obj.pos, intensity=1.0, sigma=3.5)
    
    for x in range(H):
      for y in range(W):
        tile, _ = world[(x, y)]
        if tile == 'lava':
          apply_gaussian(heatmap, (x, y), intensity=0.7, sigma=2.5)
    
    # danger index in red
    heatmap = heatmap / heatmap.max() * 255 if heatmap.max() > 0 else heatmap
    red_channel = heatmap.astype(np.uint8)
    
    # agent as a small green square
    green_channel = np.zeros_like(red_channel, dtype=np.uint8)
    player = self._env._player
    px, py = player.pos
    for dx in [-1, 0, 1]:
      for dy in [-1, 0, 1]:
        x, y = px + dx, py + dy
        if 0 <= x < H and 0 <= y < W:
          green_channel[x, y] = 255
    
    rgb_heatmap = np.stack([red_channel, green_channel, np.zeros_like(red_channel)], axis=-1)
    return rgb_heatmap

  def render(self):
    return self._env.render()
