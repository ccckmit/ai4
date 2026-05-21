//! nn/mnist/predict.rs - Placeholder for MNIST prediction

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let image_path = if args.len() > 1 { &args[1] } else { "<image_path>" };
    println!("MNIST prediction requires image processing crates (e.g. image).");
    println!("Use the Python nn/mnist/predict.py instead:");
    println!("  PYTHONPATH=. uv run python nn/mnist/predict.py {}", image_path);
}
