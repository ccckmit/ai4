// ============================================================
//  grad_check.rs  –  Numerical gradient verification
//
//  For every tested operation we:
//    1. Compute analytic grad via .backward()
//    2. Compute numerical grad via finite differences  f(x+h) - f(x-h) / 2h
//    3. Report relative error; PASS if < 1e-4
// ============================================================
use nn::tensor::{cat, SimpleRng, Tensor};
use nn::gpt::TransformerBlock;

const H: f64 = 1e-4;          // finite-diff step
const RTOL: f64 = 1e-4;       // relative tolerance for PASS

struct Report {
    passed: usize,
    failed: usize,
}

impl Report {
    fn new() -> Self { Self { passed: 0, failed: 0 } }

    fn check(&mut self, name: &str, analytic: f64, numeric: f64) {
        let denom = analytic.abs().max(numeric.abs()).max(1e-8);
        let rel = (analytic - numeric).abs() / denom;
        if rel < RTOL {
            println!("  ✓  {name:<45} analytic={analytic:+.6e}  numeric={numeric:+.6e}  rel={rel:.2e}");
            self.passed += 1;
        } else {
            println!("  ✗  {name:<45} analytic={analytic:+.6e}  numeric={numeric:+.6e}  rel={rel:.2e}  ← FAIL");
            self.failed += 1;
        }
    }

    fn summary(&self) {
        println!("\n{}", "─".repeat(70));
        println!("  PASSED: {}   FAILED: {}", self.passed, self.failed);
        println!("{}", "─".repeat(70));
        if self.failed > 0 {
            std::process::exit(1);
        }
    }
}

// Helper: build loss from tensor, run backward, return analytic grad[idx]
fn analytic_grad(
    make_input: impl Fn(Vec<f64>) -> Tensor,
    data: &[f64],
    make_loss: impl Fn(&Tensor) -> Tensor,
    idx: usize,
) -> f64 {
    let x = make_input(data.to_vec());
    let loss = make_loss(&x);
    loss.backward();
    let g = x.grad()[idx];
    g
}

// Helper: numerical grad via central difference
fn numeric_grad(
    make_input: impl Fn(Vec<f64>) -> Tensor,
    data: &[f64],
    make_loss: impl Fn(&Tensor) -> Tensor,
    idx: usize,
) -> f64 {
    let mut d_plus = data.to_vec();
    d_plus[idx] += H;
    let mut d_minus = data.to_vec();
    d_minus[idx] -= H;
    let lp = make_loss(&make_input(d_plus)).scalar_val();
    let lm = make_loss(&make_input(d_minus)).scalar_val();
    (lp - lm) / (2.0 * H)
}

// Run a quick check on a subset of indices of a given input tensor
fn check_op(
    report: &mut Report,
    label: &str,
    shape: &[usize],
    data: &[f64],
    make_loss: impl Fn(&Tensor) -> Tensor + Clone,
    check_indices: &[usize],
) {
    for &i in check_indices {
        let d2 = data.to_vec();
        let sh = shape.to_vec();
        let ml = make_loss.clone();
        let ml2 = make_loss.clone();
        let a = analytic_grad(
            |d| Tensor::from_slice(&d, &sh, true),
            &d2,
            |x| ml(x),
            i,
        );
        let n = numeric_grad(
            |d| Tensor::from_slice(&d, &sh, true),
            &d2,
            |x| ml2(x),
            i,
        );
        report.check(&format!("{} [{}]", label, i), a, n);
    }
}

fn main() {
    let mut report = Report::new();
    let mut rng = SimpleRng::new(7);

    // ─────────────────────────────────────────────────────────
    println!("\n══ 1. Element-wise operations ══");
    // ─────────────────────────────────────────────────────────

    let data3 = vec![-1.5, 0.3, 2.0, -0.7, 1.1, 0.0];
    let shape3 = [2, 3];

    check_op(&mut report, "relu", &shape3, &data3,
        |x| x.relu().sum_all(), &[0, 1, 2, 4]);

    check_op(&mut report, "tanh", &shape3, &data3,
        |x| x.tanh().sum_all(), &[0, 1, 3, 5]);

    check_op(&mut report, "pow(3)", &shape3, &data3,
        |x| x.pow(3.0).sum_all(), &[0, 2, 4]);

    check_op(&mut report, "mul_scalar(2.5)", &shape3, &data3,
        |x| x.mul_scalar(2.5).sum_all(), &[0, 3]);

    check_op(&mut report, "add_scalar(-1)", &shape3, &data3,
        |x| x.add_scalar(-1.0).sum_all(), &[1, 4]);

    // neg
    check_op(&mut report, "neg", &shape3, &data3,
        |x| x.neg().sum_all(), &[0, 5]);

    // ─────────────────────────────────────────────────────────
    println!("\n══ 2. Reduction & softmax ══");
    // ─────────────────────────────────────────────────────────

    let data4 = vec![1.0, -0.5, 0.3, 2.0, 0.1, -1.2];
    let shape4 = [2, 3];

    check_op(&mut report, "sum_all", &shape4, &data4,
        |x| x.sum_all(), &[0, 2, 5]);

    check_op(&mut report, "mean_all", &shape4, &data4,
        |x| x.mean_all(), &[1, 3]);

    check_op(&mut report, "softmax(axis=1) then weighted sum", &shape4, &data4,
        |x| {
            // weight the softmax output so the loss isn't constant
            let weights = Tensor::from_slice(&[1.0f64,2.0,3.0,0.5,1.5,2.5], &[2,3], false);
            x.softmax(1).mul(&weights).sum_all()
        }, &[0, 1, 2, 3, 4, 5]);

    // 4-D softmax (attention-like)
    let data_attn: Vec<f64> = (0..24).map(|i| i as f64 * 0.1 - 1.2).collect();
    let attn_weights = Tensor::from_slice(
        &(0..24).map(|i| (i as f64) * 0.05).collect::<Vec<_>>(), &[2,2,2,3], false);
    check_op(&mut report, "softmax 4D [2,2,2,3] axis=3 weighted", &[2,2,2,3], &data_attn,
        move |x| x.softmax(3).mul(&attn_weights).sum_all(), &[0, 5, 11, 23]);

    // ─────────────────────────────────────────────────────────
    println!("\n══ 3. matmul ══");
    // ─────────────────────────────────────────────────────────

    let a_data: Vec<f64> = (0..6).map(|i| (i as f64 - 2.5) * 0.5).collect();
    let b_data: Vec<f64> = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, 1.0, -1.1, 1.2];
    let b_t = Tensor::from_slice(&b_data, &[3, 4], false);

    check_op(&mut report, "matmul [2,3] x [3,4] grad-A", &[2, 3], &a_data,
        |x| x.matmul(&b_t).sum_all(), &[0, 2, 5]);

    let a_t = Tensor::from_slice(&a_data, &[2, 3], false);
    check_op(&mut report, "matmul [2,3] x [3,4] grad-B", &[3, 4], &b_data,
        |x| a_t.matmul(x).sum_all(), &[0, 4, 11]);

    // batched matmul
    let ba: Vec<f64> = (0..12).map(|i| i as f64 * 0.1).collect();
    let bb: Vec<f64> = (0..12).map(|i| (11 - i) as f64 * 0.1).collect();
    let bb_t = Tensor::from_slice(&bb, &[2, 3, 2], false);
    check_op(&mut report, "batched matmul [2,2,3] grad", &[2,2,3], &ba,
        |x| x.matmul(&bb_t).sum_all(), &[0, 5, 11]);

    // ─────────────────────────────────────────────────────────
    println!("\n══ 4. transpose ══");
    // ─────────────────────────────────────────────────────────

    let td: Vec<f64> = (0..24).map(|i| i as f64 * 0.5 - 6.0).collect();
    check_op(&mut report, "transpose [2,3,4] (0,1)", &[2,3,4], &td,
        |x| x.transpose(0, 1).sum_all(), &[0, 7, 23]);
    check_op(&mut report, "transpose [2,3,4] (1,2)", &[2,3,4], &td,
        |x| x.transpose(1, 2).sum_all(), &[1, 8, 22]);
    check_op(&mut report, "transpose [2,3,4] (0,2)", &[2,3,4], &td,
        |x| x.transpose(0, 2).sum_all(), &[3, 15]);

    // ─────────────────────────────────────────────────────────
    println!("\n══ 5. reshape ══");
    // ─────────────────────────────────────────────────────────

    check_op(&mut report, "reshape [2,3]->[6,1]", &[2,3], &data3,
        |x| x.reshape(vec![6, 1]).sum_all(), &[0, 4]);

    // ─────────────────────────────────────────────────────────
    println!("\n══ 6. cat ══");
    // ─────────────────────────────────────────────────────────

    let c1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let c2: Vec<f64> = vec![5.0, 6.0, 7.0, 8.0];
    let t2 = Tensor::from_slice(&c2, &[2, 2], false);
    check_op(&mut report, "cat axis=0, grad of t1", &[2,2], &c1,
        |x| cat(&[x.clone(), t2.clone()], 0).sum_all(), &[0, 3]);

    let t1f = Tensor::from_slice(&c1, &[2, 2], false);
    check_op(&mut report, "cat axis=1, grad of t2", &[2,2], &c2,
        |x| cat(&[t1f.clone(), x.clone()], 1).sum_all(), &[0, 3]);

    // ─────────────────────────────────────────────────────────
    println!("\n══ 7. rms_norm ══");
    // ─────────────────────────────────────────────────────────

    let rn_data: Vec<f64> = (0..12).map(|i| (i as f64 - 5.5) * 0.7).collect();
    check_op(&mut report, "rms_norm [3,4]", &[3,4], &rn_data,
        |x| x.rms_norm().sum_all(), &[0, 3, 7, 11]);

    check_op(&mut report, "rms_norm [2,2,3]", &[2,2,3], &rn_data[..12].to_vec(),
        |x| x.rms_norm().sum_all(), &[0, 5, 11]);

    // ─────────────────────────────────────────────────────────
    println!("\n══ 8. cross_entropy ══");
    // ─────────────────────────────────────────────────────────

    let ce_data: Vec<f64> = vec![1.0, 2.0, 0.5, -1.0, 0.5, 3.0];
    check_op(&mut report, "cross_entropy [2,3] targets=[1,2]", &[2,3], &ce_data,
        |x| x.cross_entropy(&[1, 2]), &[0, 1, 2, 3, 4, 5]);

    // ─────────────────────────────────────────────────────────
    println!("\n══ 9. Linear layer ══");
    // ─────────────────────────────────────────────────────────
    {
        // grad w.r.t. input x: freeze weight, differentiate x
        let lin_w: Vec<f64> = (0..12).map(|i| (i as f64 - 5.5) * 0.08).collect();
        let x_data: Vec<f64> = (0..8).map(|i| (i as f64 - 3.5) * 0.3).collect();
        let w_t = Tensor::from_slice(&lin_w, &[3, 4], false); // weight frozen

        check_op(&mut report, "Linear forward grad-x [2,4]->[2,3]", &[2,4], &x_data,
            move |x| {
                let wt = w_t.transpose(0, 1);
                x.matmul(&wt).sum_all()
            }, &[0, 3, 7]);

        // grad w.r.t. weight W: freeze x, differentiate W
        let x_t = Tensor::from_slice(&x_data, &[2, 4], false);
        check_op(&mut report, "Linear forward grad-W [3,4]", &[3,4], &lin_w,
            move |w| {
                let wt = w.transpose(0, 1);
                x_t.matmul(&wt).sum_all()
            }, &[0, 5, 11]);
    }

    // ─────────────────────────────────────────────────────────
    println!("\n══ 10. Chained computation ══");
    // ─────────────────────────────────────────────────────────

    let chain_data: Vec<f64> = vec![0.5, -1.0, 2.0, 0.1];
    check_op(&mut report, "chain: relu→pow(2)→mean", &[2,2], &chain_data,
        |x| x.relu().pow(2.0).mean_all(), &[0, 1, 2, 3]);

    check_op(&mut report, "chain: tanh→mul_scalar→sum", &[2,2], &chain_data,
        |x| x.tanh().mul_scalar(3.0).sum_all(), &[0, 2]);

    // matmul → relu → sum (tests full non-linear path)
    let mat_data: Vec<f64> = vec![1.0, -0.5, 0.3, -0.8];
    let other_t = Tensor::from_slice(&[0.2f64, -0.3, 0.5, 0.1], &[2, 2], false);
    check_op(&mut report, "chain: matmul→relu→sum", &[2,2], &mat_data,
        |x| x.matmul(&other_t).relu().sum_all(), &[0, 1, 2, 3]);

    // ─────────────────────────────────────────────────────────
    println!("\n══ 11. Attention sub-components ══");
    // ─────────────────────────────────────────────────────────
    {
        // Q·K^T / sqrt(d) → softmax → ·V  with small tensors
        let b = 1usize; let nh = 2usize; let t = 3usize; let hd = 4usize;
        let q_data: Vec<f64> = (0..b*nh*t*hd).map(|i| (i as f64 - 12.0) * 0.1).collect();
        let k_data: Vec<f64> = (0..b*nh*t*hd).map(|i| (12.0 - i as f64) * 0.1).collect();
        let v_data: Vec<f64> = (0..b*nh*t*hd).map(|i| i as f64 * 0.05).collect();
        let scale = 1.0 / (hd as f64).sqrt();

        let k_t = Tensor::from_slice(&k_data, &[b,nh,t,hd], false);
        let v_t = Tensor::from_slice(&v_data, &[b,nh,t,hd], false);
        check_op(&mut report, "attn Q grad [1,2,3,4]", &[b,nh,t,hd], &q_data,
            move |q| {
                let kt = k_t.transpose(2, 3);
                let aw = q.matmul(&kt).mul_scalar(scale).softmax(3);
                aw.matmul(&v_t).sum_all()
            }, &[0, 5, 11, 23]);

        let q_t2 = Tensor::from_slice(&q_data, &[b,nh,t,hd], false);
        let v_t2 = Tensor::from_slice(&v_data, &[b,nh,t,hd], false);
        check_op(&mut report, "attn K grad [1,2,3,4]", &[b,nh,t,hd], &k_data,
            move |k| {
                let kt = k.transpose(2, 3);
                let aw = q_t2.matmul(&kt).mul_scalar(scale).softmax(3);
                aw.matmul(&v_t2).sum_all()
            }, &[0, 5, 23]);

        let q_t3 = Tensor::from_slice(&q_data, &[b,nh,t,hd], false);
        let k_t3 = Tensor::from_slice(&k_data, &[b,nh,t,hd], false);
        check_op(&mut report, "attn V grad [1,2,3,4]", &[b,nh,t,hd], &v_data,
            move |v| {
                let kt = k_t3.transpose(2, 3);
                let aw = q_t3.matmul(&kt).mul_scalar(scale).softmax(3);
                aw.matmul(v).sum_all()
            }, &[0, 8, 23]);
    }

    // ─────────────────────────────────────────────────────────
    println!("\n══ 12. TransformerBlock end-to-end ══");
    // ─────────────────────────────────────────────────────────
    {
        let block = TransformerBlock::new(8, 2, &mut rng);
        let x_data: Vec<f64> = (0..1*3*8).map(|i| (i as f64 - 12.0) * 0.2).collect();
        let targets = vec![1usize, 0, 2];
        let vocab = 4usize;
        let head_w: Vec<f64> = (0..vocab*8).map(|i| i as f64 * 0.05 - 1.0).collect();

        // Manual grad check without check_op (block isn't Clone)
        let check_indices = [0usize, 5, 11, 23];
        for &i in &check_indices {
            // analytic
            let x = Tensor::from_slice(&x_data, &[1, 3, 8], true);
            let ht = Tensor::from_slice(&head_w, &[vocab, 8], false).transpose(0, 1);
            let (out, _) = block.forward(&x, None);
            let logits = out.reshape(vec![3, 8]).matmul(&ht);
            let loss = logits.cross_entropy(&targets);
            loss.backward();
            let analytic = x.grad()[i];

            // numeric
            let mut dp = x_data.clone(); dp[i] += H;
            let mut dm = x_data.clone(); dm[i] -= H;
            let xp = Tensor::from_slice(&dp, &[1, 3, 8], false);
            let xm = Tensor::from_slice(&dm, &[1, 3, 8], false);
            let htp = Tensor::from_slice(&head_w, &[vocab, 8], false).transpose(0, 1);
            let htm = Tensor::from_slice(&head_w, &[vocab, 8], false).transpose(0, 1);
            let (op, _) = block.forward(&xp, None);
            let lp = op.reshape(vec![3, 8]).matmul(&htp).cross_entropy(&targets).scalar_val();
            let (om, _) = block.forward(&xm, None);
            let lm = om.reshape(vec![3, 8]).matmul(&htm).cross_entropy(&targets).scalar_val();
            let numeric = (lp - lm) / (2.0 * H);

            report.check(&format!("TransformerBlock grad-x [{}]", i), analytic, numeric);
        }
    }

    // ─────────────────────────────────────────────────────────
    println!("\n══ 13. broadcast operations ══");
    // ─────────────────────────────────────────────────────────

    let big: Vec<f64> = (0..6).map(|i| i as f64 * 0.5).collect();
    let small = Tensor::from_slice(&[1.0f64, -0.5, 0.3], &[3], false);
    check_op(&mut report, "broadcast add [2,3]+[3]", &[2,3], &big,
        move |x| x.add(&small).sum_all(), &[0, 2, 5]);

    let big2: Vec<f64> = (0..6).map(|i| (i as f64 + 1.0) * 0.3).collect();
    let small2 = Tensor::from_slice(&[0.5f64, -1.0], &[2, 1], false);
    check_op(&mut report, "broadcast mul [2,3]*[2,1]", &[2,3], &big2,
        move |x| x.mul(&small2).sum_all(), &[0, 3, 5]);

    // ─────────────────────────────────────────────────────────
    report.summary();
}
