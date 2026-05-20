import * as fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { Tensor } from './tensor';
import { Adam } from './optim';
import { GPT } from './gpt';
import { train_model, generate_samples } from './chargpt';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function main() {
  const dataPath = join(__dirname, '..', 'data', 'input.txt');
  let docs: string[] = [];

  if (fs.existsSync(dataPath)) {
    const content = fs.readFileSync(dataPath, 'utf-8');
    docs = content.split('\n').filter((line: string) => line.trim());
    console.log(`Loaded ${docs.length} names from data/input.txt`);
  } else {
    docs = [
      'John', 'Mary', 'David', 'Sarah', 'Michael',
      'Emma', 'William', 'Olivia', 'James', 'Ava'
    ];
    console.log('data/input.txt not found, using default names');
  }

  console.log(`num docs: ${docs.length}`);

  const allChars = Array.from(new Set(docs.join('').split('')));
  const uchars = allChars.sort();
  const BOS = uchars.length;
  const vocab_size = uchars.length + 1;
  console.log(`vocab size: ${vocab_size}`);

  const block_size = 16;
  const model = new GPT(vocab_size, block_size, 1, 16, 4);
  console.log(`num params: ${model.parameters().length}`);

  const optimizer = new Adam(model.parameters(), 0.01);

  train_model(
    model, optimizer, docs, uchars,
    BOS, block_size, 1000
  );

  generate_samples(
    model, uchars, BOS, vocab_size,
    block_size, 20, 0.5
  );
}

main();