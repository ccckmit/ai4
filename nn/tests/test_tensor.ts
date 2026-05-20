import { Tensor } from '../tensor';

const a = Tensor.from([[1, 2], [3, 4]], true);
const b = Tensor.from([[5, 6], [7, 8]], true);
const sum = a.add(b);
console.log('add:', sum.data);
sum.backward();
console.log('grad a:', a.grad);

const t3 = Tensor.from([[1], [2]]);
const t4 = Tensor.from([[3, 4]]);
const mm = t3.matmul(t4);
console.log('matmul (1x2 @ 2x2):', mm.data);

const t5 = Tensor.from([[-1, 2, -3]]);
const r = t5.relu();
console.log('relu:', r.data);

const tsum = Tensor.from([[1, 2, 3]]);
const s = tsum.sum();
console.log('sum:', s.data);

console.log('nn/tensor tests passed');