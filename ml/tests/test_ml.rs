//! ml/tests/test_ml.rs - ML toolkit tests.

use crate::ml::{LinearRegression, LogisticRegression, KMeans, PCA, StandardScaler, accuracy_score, mean_squared_error, r2_score};

#[test]
fn test_linear_regression() {
    let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

    let mut model = LinearRegression::new(0.01, 1000);
    model.fit(&x, &y);

    let pred = model.predict(&vec![vec![6.0]]);
    assert!((pred[0] - 12.0).abs() < 1.0);
}

#[test]
fn test_accuracy() {
    let y_true = vec![0, 1, 2, 0, 1, 2];
    let y_pred = vec![0, 1, 2, 0, 0, 2];
    assert_eq!(accuracy_score(&y_true, &y_pred), 5.0 / 6.0);
}

#[test]
fn test_logistic_regression() {
    let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0], vec![6.0]];
    let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let mut model = LogisticRegression::new(0.1, 1000);
    model.fit(&x, &y);

    let pred = model.predict(&x);
    let acc = accuracy_score(&y.iter().map(|&x| x as i32).collect::<Vec<_>>(), &pred);
    assert!(acc > 0.7);
}

#[test]
fn test_kmeans() {
    let mut x1 = Vec::new();
    let mut x2 = Vec::new();

    for _ in 0..30 {
        x1.push(vec![rand::random::<f64>() * 2.0 + 2.0, rand::random::<f64>() * 2.0 + 2.0]);
        x2.push(vec![rand::random::<f64>() * 2.0 - 2.0, rand::random::<f64>() * 2.0 - 2.0]);
    }

    let mut x = x1;
    x.extend(x2);

    let mut kmeans = KMeans::new(2, 300, 5);
    kmeans.fit(&x);

    let labels = kmeans.predict(&x);
    assert_eq!(labels.len(), x.len());
}

#[test]
fn test_pca() {
    let mut x = Vec::new();
    for _ in 0..50 {
        x.push(vec![rand::random(), rand::random(), rand::random(), rand::random()]);
    }

    let mut pca = PCA::new(Some(2));
    let x_transformed = pca.fit_transform(&x);

    assert_eq!(x_transformed.len(), 50);
    assert_eq!(x_transformed[0].len(), 2);
}

#[test]
fn test_scaler() {
    let x = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

    let mut scaler = StandardScaler::new();
    let x_scaled = scaler.fit_transform(&x);

    for col in 0..2 {
        let mean: f64 = x_scaled.iter().map(|row| row[col]).sum::<f64>() / x_scaled.len() as f64;
        assert!((mean - 0.0).abs() < 1e-10);
    }
}

#[test]
fn test_mean_squared_error() {
    let y_true = vec![1.0, 2.0, 3.0, 4.0];
    let y_pred = vec![1.1, 2.1, 2.9, 4.1];
    let mse = mean_squared_error(&y_true, &y_pred);
    assert!(mse < 0.1);
}

#[test]
fn test_r2_score() {
    let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = vec![1.1, 2.1, 2.9, 4.1, 4.9];
    let r2 = r2_score(&y_true, &y_pred);
    assert!(r2 > 0.95);
}