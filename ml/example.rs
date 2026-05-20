//! ml/example.rs - Example of using ML models

use ml::linear_models::{LinearRegression, LogisticRegression};
use ml::metrics::accuracy_score;

fn main() {
    println!("=== Linear Regression Example ===");
    
    // Training data: y = 2x + 1
    let x = vec![
        vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0],
    ];
    let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];
    
    let mut model = LinearRegression::new(0.1, 1000);
    model.fit(&x, &y);
    
    let pred = model.predict(&x);
    for (x_val, y_pred) in x.iter().zip(&pred) {
        println!("x={:.1}, predicted={:.2}, actual={:.1}", x_val[0], y_pred, 2.0 * x_val[0] + 1.0);
    }
    
    println!("\n=== Logistic Regression Example ===");
    
    // Binary classification
    let x = vec![
        vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0],
    ];
    let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    
    let mut model = LogisticRegression::new(0.1, 1000);
    model.fit(&x, &y);
    
    let pred = model.predict(&x);
    let y_int: Vec<i32> = y.iter().map(|&v| v as i32).collect();
    let acc = accuracy_score(&y_int, &pred);
    println!("Accuracy: {:.1}%", acc * 100.0);
}