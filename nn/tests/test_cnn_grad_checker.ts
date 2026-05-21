import { Tensor, Conv2d, MaxPool2d, Flatten } from '../index';

function numerical_grad(f: (t: Tensor[]) => number, inputs: Tensor[], eps: number = 1e-4): number[][] {
  const grads: number[][] = [];
  for (let i = 0; i < inputs.length; i++) {
    const tensor = inputs[i];
    if (!tensor) {
      grads.push([]);
      continue;
    }
    const grad = new Array(tensor.data.length).fill(0);
    for (let j = 0; j < tensor.data.length; j++) {
      const orig = tensor.data[j];
      
      tensor.data[j] = orig + eps;
      const v_plus = f(inputs);
      
      tensor.data[j] = orig - eps;
      const v_minus = f(inputs);
      
      tensor.data[j] = orig;
      
      grad[j] = (v_plus - v_minus) / (2 * eps);
    }
    grads.push(grad);
  }
  return grads;
}

function check_grad(name: string, f: (t: Tensor[]) => Tensor, inputs: Tensor[]) {
  const out = f(inputs);
  const loss = out.sum();
  
  inputs.forEach(t => { if (t) t.grad.fill(0); });
  loss.backward();
  
  const analytic_grads = inputs.map(t => t ? t.grad : []);
  
  const num_grads = numerical_grad((tensors) => {
    return f(tensors).sum().data[0];
  }, inputs);
  
  let max_diff = 0;
  for (let i = 0; i < inputs.length; i++) {
    if (!inputs[i]) continue;
    for (let j = 0; j < inputs[i].data.length; j++) {
      const diff = Math.abs(analytic_grads[i][j] - num_grads[i][j]);
      if (diff > max_diff) max_diff = diff;
    }
  }
  
  console.log(`${name}: max difference = ${max_diff}`);
}

console.log("=== MaxPool2d ===");
const pool = new MaxPool2d(2);
const xPool = new Tensor(
  new Array(1*1*4*4).fill(0).map(()=>Math.random()), 
  [1, 1, 4, 4], 
  true
);
check_grad("MaxPool2d", (inputs) => pool.forward(inputs[0]), [xPool]);

console.log("=== Conv2d ===");
const conv = new Conv2d(1, 1, 3, 1, 0, true);
const xConv = new Tensor(
  new Array(1*1*4*4).fill(0).map(()=>Math.random()), 
  [1, 1, 4, 4], 
  true
);
check_grad("Conv2d(Input)", (inputs) => {
  const c = new Conv2d(1, 1, 3, 1, 0, true);
  c.weight = inputs[1];
  c.bias = inputs[2];
  return c.forward(inputs[0]);
}, [xConv, conv.weight, conv.bias as Tensor]);

console.log("=== Flatten ===");
const flatten = new Flatten();
const xFlat = new Tensor(
  new Array(1*3*2*2).fill(0).map(()=>Math.random()), 
  [1, 3, 2, 2], 
  true
);
check_grad("Flatten", (inputs) => flatten.forward(inputs[0]), [xFlat]);
