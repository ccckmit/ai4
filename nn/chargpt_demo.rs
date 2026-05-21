use ai4::nn::chargpt::{generate_samples, train_model};
use ai4::nn::gpt::GPT;
use ai4::nn::Adam;
use ai4::nn::tensor::SimpleRng;

fn shuffle<T>(v: &mut [T], rng: &mut SimpleRng) {
    for i in (1..v.len()).rev() {
        let j = rng.next_usize(i + 1);
        v.swap(i, j);
    }
}

fn main() {
    let mut rng = SimpleRng::new(42);

    let docs: Vec<String> = std::fs::read_to_string("data/input.txt")
        .expect("Failed to read data/input.txt")
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    println!("num docs: {}", docs.len());

    let mut all_chars: Vec<char> = docs.join("").chars().collect::<std::collections::HashSet<_>>().into_iter().collect();
    all_chars.sort();
    let uchars = all_chars;
    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("vocab size: {}", vocab_size);

    let mut docs = docs;
    shuffle(&mut docs, &mut rng);

    let block_size = 16usize;
    let model = GPT::new(vocab_size, block_size, 1, 16, 4, &mut rng);
    println!("num params: {}", model.parameters().len());

    let params = model.parameters();
    let mut optimizer = Adam::new(params, 0.01);

    train_model(&model, &mut optimizer, &docs, &uchars, bos, block_size, 1000);
    generate_samples(&model, &uchars, bos, vocab_size, block_size, 20, 0.5);
}