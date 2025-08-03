import numpy as np
import os
import re

from collections import defaultdict
from PIL import Image

import embodied

class ImageUtil:
    def create_if_not_there(self, folders):
        for folder in folders:
            if os.path.exists(folder) is False:
                os.mkdir(folder)

    def __init__(self, base_folder, experiment_label=None):
        print('ImageUtil::__init__ base_folder = ', base_folder, ', experiment_label = ', experiment_label)
        images_folder = os.path.join(base_folder, experiment_label)
        self.create_if_not_there([base_folder, images_folder])

        self.actual_image_folder = os.path.join(images_folder, 'actual')
        self.fwd_image_folder = os.path.join(images_folder, 'fwd_decoded')
        self.rev_image_folder = os.path.join(images_folder, 'rev_decoded')
        self.comp_image_folder = os.path.join(images_folder, 'comparison_grids')
        self.create_if_not_there([self.actual_image_folder, self.fwd_image_folder, self.rev_image_folder, self.comp_image_folder])

    def normalize_image(self, image):
        return image/255.0

    def denormalize_image(self, image):
        return np.clip(255 * image, 0, 255).astype(np.uint8)

    def print_image(self, image_mat, image_folder, image_name, is_normalized=True):
        if is_normalized:
            image_mat = self.denormalize_image(image_mat)
        image_path = os.path.join(image_folder, image_name)
        img = Image.fromarray(image_mat, 'RGB')
        img.save(image_path)

    def print_all_images(self, actual_image, fwd_pred_image, rev_pred_images, name_prefix, name_postfix=None):
        # print individual images
        self.print_image(actual_image, self.actual_image_folder, name_prefix + '_actual.png', False)
        self.print_image(fwd_pred_image, self.fwd_image_folder, name_prefix + '_fwd_decoded.png')

        if len(rev_pred_images.shape) > 1:
            for i in range(len(rev_pred_images)):
                self.print_image(rev_pred_images[i], self.rev_image_folder, name_prefix + '_rev_decoded_' + str(i) + '.png')
        else: 
            self.print_image(rev_pred_images[0], self.rev_image_folder, name_prefix + '_rev_decoded.png')

        # print comparison grid
        normalized_actual_image = self.normalize_image(actual_image)
        combined_rev_pred_image = np.concatenate([image for image in rev_pred_images], axis=1)
        combined_images = np.concatenate([normalized_actual_image, fwd_pred_image, combined_rev_pred_image], axis=1)

        baseline = np.tile(normalized_actual_image, (1, 2 + len(rev_pred_images), 1))
        diff = combined_images - baseline
        image_comparison_grid = np.concatenate([combined_images, diff], axis=0)
        self.print_image(image_comparison_grid, self.comp_image_folder, name_prefix + '_comparison.png')

class TrajectoryCache:
    def __init__(self, keys) -> None:
        if not isinstance(keys, list):
            raise TypeError('Keys need to be supplied as a list')

        self.keys_of_interest = keys
        self.observations = {k:None for k in self.keys_of_interest}

    def cache_partial_obs(self, tran, _):
        for k in self.keys_of_interest:
            if isinstance(tran[k], np.ndarray):
                entry = tran[k][np.newaxis, :]
            else:
                entry = np.array([tran[k]])
            self.observations[k] = entry if self.observations[k] is None else np.concatenate([self.observations[k], entry], axis = 0)

    def get_actions(self):
        return self.observations['action']

    def get_action_at(self, i):
        return self.observations['action'][i]

    def get_image_at(self, i):
        return self.observations['image'][i]
    
    def get_images(self):
        return self.observations['image']
    
    def get_obs_at(self, i):
        return {k:np.array([self.observations[k][i]]) for k in self.keys_of_interest}

    def get_rev_obss_between(self, start_index, stop_index):
        observations = {}
        for k in self.keys_of_interest:
            observations[k] = np.flip(np.array([self.observations[k][start_index:stop_index]]), axis = 1)
            print('TrajectoryCache::get_rev_obss_between observations[', str(k), '].shape = ', observations[k].shape)
        return observations

    def get_cache_count(self):
        return len(self.get_actions())

class ExperimentTracker:
    def __init__(self, args) -> None:
        self.args = args

    def setup_tracking(self, make_logger):
        self.logger = make_logger()
        self.logdir = embodied.Path(self.args.logdir)
        self.logdir.mkdir()
        print('Logdir', self.logdir)
        self.step = self.logger.step
        self.usage = embodied.Usage(**self.args.usage)
        self.agg = embodied.Agg()
        self.epstats = embodied.Agg()
        self.episodes = defaultdict(embodied.Agg)
        self.should_log = embodied.when.Clock(self.args.log_every)
        self.policy_fps = embodied.FPS()
        self.is_first_episode_complete = False

    @embodied.timer.section('log_step')
    def log_step(self, tran, worker):
        episode = self.episodes[worker]
        episode.add('score', tran['reward'], agg='sum')
        episode.add('length', 1, agg='sum')
        episode.add('rewards', tran['reward'], agg='stack')

        if tran['is_first']:
            episode.reset()

        if worker < self.args.log_video_streams:
            for key in self.args.log_keys_video:
                if key in tran:
                    episode.add(f'policy_{key}', tran[key], agg='stack')
        for key, value in tran.items():
            if re.match(self.args.log_keys_sum, key):
                episode.add(key, value, agg='sum')
            if re.match(self.args.log_keys_avg, key):
                episode.add(key, value, agg='avg')
            if re.match(self.args.log_keys_max, key):
                episode.add(key, value, agg='max')

        if tran['is_last']:
            result = episode.result()
            self.is_first_episode_complete = True

            self.logger.add({
                'score': result.pop('score'),
                'length': result.pop('length') - 1,
            }, prefix='episode')
            rew = result.pop('rewards')
            if len(rew) > 1:
                result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
            self.epstats.add(result)

    def log_stats(self):
        if self.should_log(self.step):
            self.logger.add(self.agg.result())
            self.logger.add(self.epstats.result(), prefix='epstats')
            self.logger.add(embodied.timer.stats(), prefix='timer')
            self.logger.add(self.usage.stats(), prefix='usage')
            self.logger.add({'fps/policy': self.policy_fps.result()})
            self.logger.write()

    def experiment_complete(self):
        self.logger.close()