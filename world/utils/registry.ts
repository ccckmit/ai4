import { Env } from '../core';
import { FrozenLakeEnv } from '../envs/frozen_lake';
import { CartPoleEnv } from '../envs/cartpole';

type EnvCtor = new (...args: unknown[]) => Env;

const REGISTRY: Record<string, EnvCtor> = {};

export function register(id: string, cls: EnvCtor): void {
  REGISTRY[id] = cls;
}

export function registry(): Record<string, EnvCtor> {
  return { ...REGISTRY };
}

export function make(id: string, ...args: unknown[]): Env {
  if (!REGISTRY[id]) {
    throw new Error(`Unknown env: ${id}. Available: ${Object.keys(REGISTRY).join(', ')}`);
  }
  return new REGISTRY[id](...args);
}

register('FrozenLake-v0', FrozenLakeEnv as unknown as EnvCtor);
register('FrozenLake-v1', class extends FrozenLakeEnv {
  constructor() { super('4x4', undefined, true); }
} as unknown as EnvCtor);
register('FrozenLake8x8-v1', class extends FrozenLakeEnv {
  constructor() { super('8x8', undefined, true); }
} as unknown as EnvCtor);
register('CartPole-v1', CartPoleEnv as unknown as EnvCtor);