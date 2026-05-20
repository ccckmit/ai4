import { Env, StepResult } from '../core';
import { Discrete } from '../spaces/discrete';

const MAPS: Record<string, string[]> = {
  '4x4': ['SFFF', 'FHFH', 'FFFH', 'HFFG'],
  '8x8': [
    'SFFFFFFF', 'FFFFFFFF', 'FFFHFFFF', 'FFFFFHFF',
    'FFFHFFFF', 'FHHFFFHF', 'FHFFHFHF', 'FFFHFFFG',
  ],
};

const ACTIONS: [number, number][] = [
  [0, -1],  // LEFT
  [1,  0],  // DOWN
  [0,  1],  // RIGHT
  [-1, 0],  // UP
];

const ACTION_NAMES = ['LEFT', 'DOWN', 'RIGHT', 'UP'];

const ANSI: Record<string, string> = {
  S: '\x1b[94mS\x1b[0m',
  F: '\x1b[97mF\x1b[0m',
  H: '\x1b[91mH\x1b[0m',
  G: '\x1b[92mG\x1b[0m',
  A: '\x1b[93mA\x1b[0m',
};

export class FrozenLakeEnv extends Env<number, number> {
  private _desc: string[][] = [];
  private _nrow = 0;
  private _ncol = 0;
  private _is_slippery = true;
  private _start: [number, number] = [0, 0];
  private _goal: [number, number] = [0, 0];
  private _agent_pos: [number, number] = [0, 0];
  private _max_steps = 100;
  private _observation_space: Discrete;
  private _action_space: Discrete;

  constructor(
    map_name = '4x4',
    custom_map?: string[],
    is_slippery = true,
    max_steps?: number
  ) {
    super();
    const desc = custom_map ?? MAPS[map_name] ?? MAPS['4x4'];
    this._desc = desc.map(row => row.split(''));
    this._nrow = this._desc.length;
    this._ncol = this._desc[0].length;
    this._is_slippery = is_slippery;
    this._start = this._find('S');
    this._goal = this._find('G');
    this._max_steps = max_steps ?? 100 * this._nrow;
    this._agent_pos = [...this._start] as [number, number];
    this._observation_space = new Discrete(this._nrow * this._ncol);
    this._action_space = new Discrete(4);
  }

  get observationSpace(): Discrete { return this._observation_space; }
  get actionSpace(): Discrete { return this._action_space; }

  reset(options?: { seed?: number }): { observation: number; info: Record<string, unknown> } {
    this.initRNG(options?.seed);
    this._agent_pos = [...this._start] as [number, number];
    this._steps = 0;
    const obs = this._posToState(this._agent_pos);
    return { observation: obs, info: { pos: this._agent_pos } };
  }

  step(action: number): StepResult<number> {
    if (!this._action_space.contains(action)) {
      throw new Error(`Invalid action ${action}`);
    }
    this._steps++;

    if (this._is_slippery) {
      const candidates = [(action - 1) % 4, action, (action + 1) % 4];
      action = this._rng.choice(candidates as number[]);
    }

    const [dr, dc] = ACTIONS[action];
    let nr = this._agent_pos[0] + dr;
    let nc = this._agent_pos[1] + dc;
    nr = Math.max(0, Math.min(nr, this._nrow - 1));
    nc = Math.max(0, Math.min(nc, this._ncol - 1));
    this._agent_pos = [nr, nc];

    const cell = this._desc[nr][nc];
    const terminated = cell === 'H' || cell === 'G';
    const truncated = !terminated && this._steps >= this._max_steps;
    const reward = cell === 'G' ? 1.0 : 0.0;
    const obs = this._posToState(this._agent_pos);

    return new StepResult(obs, reward, terminated, truncated, {
      pos: this._agent_pos,
      cell,
      action_taken: ACTION_NAMES[action],
      steps: this._steps,
    });
  }

  render(): string {
    const [ar, ac] = this._agent_pos;
    const lines: string[] = [];
    for (let r = 0; r < this._nrow; r++) {
      let line = '';
      for (let c = 0; c < this._ncol; c++) {
        if (r === ar && c === ac) {
          line += ANSI['A'];
        } else {
          line += ANSI[this._desc[r][c]] ?? this._desc[r][c];
        }
      }
      lines.push(line);
    }
    const out = lines.join('\n');
    console.log(out);
    return out;
  }

  private _posToState(pos: [number, number]): number {
    return pos[0] * this._ncol + pos[1];
  }

  private _find(char: string): [number, number] {
    for (let r = 0; r < this._nrow; r++) {
      for (let c = 0; c < this._ncol; c++) {
        if (this._desc[r][c] === char) return [r, c];
      }
    }
    throw new Error(`Char '${char}' not found`);
  }
}