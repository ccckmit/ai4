export async function predict(imagePath: string) {
  console.log('MNIST prediction requires the nn/datasets.ts MNIST loader.');
  console.log('Use the Python nn/mnist/predict.py instead:');
  console.log('  PYTHONPATH=. uv run python nn/mnist/predict.py', imagePath);
}