import { Env, StepResult } from '../core';

export class RecordEpisodeWrapper<ObsType, ActType> extends Env<ObsType, ActType> {
  private _env: Env<ObsType, ActType>;
  episode_reward = 0;
  episode_length = 0;
  episode_stats: { reward: number; length: number }[] = [];
  current_episode: { rewards: number[]; lengths: number[] } = { rewards: [], lengths: [] };

  get observationSpace() { return this._env.observationSpace; }
  get actionSpace() { return this._env.actionSpace; }

  constructor(env: Env<ObsType, ActType>) {
    super();
    this._env = env;
  }

  reset(options?: { seed?: number }) {
    this.episode_reward = 0;
    this.episode_length = 0;
    this.current_episode = { rewards: [], lengths: [] };
    return this._env.reset(options);
  }

  step(action: ActType): StepResult<ObsType> {
    const result = this._env.step(action);
    this.episode_reward += result.reward;
    this.episode_length++;
    this.current_episode.rewards.push(result.reward);
    this.current_episode.lengths.push(this.episode_length);

    if (result.done) {
      this.episode_stats.push({
        reward: this.episode_reward,
        length: this.episode_length,
      });
    }
    return result;
  }

  summary(): { episodes: number; mean_reward: number; max_reward: number; mean_length: number } {
    const stats = this.episode_stats;
    if (stats.length === 0) {
      return { episodes: 0, mean_reward: 0, max_reward: 0, mean_length: 0 };
    }
    const rewards = stats.map(s => s.reward);
    const lengths = stats.map(s => s.length);
    return {
      episodes: stats.length,
      mean_reward: rewards.reduce((a, b) => a + b, 0) / rewards.length,
      max_reward: Math.max(...rewards),
      mean_length: lengths.reduce((a, b) => a + b, 0) / lengths.length,
    };
  }

  render() { return this._env.render(); }
  close() { this._env.close(); }
}