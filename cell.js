function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
  const s = sigmoid(x);
  return s * (1 - s);
}

let weight = Math.random();
let bias = Math.random();
let input = 0.1;
let target = 1.0;
let lr = 0.1;

function forward(x) {
  const z = weight * x + bias;
  return sigmoid(z);
}

for (let i = 0; i < 1000; i++) {
  const z = weight * input + bias;
  const output = sigmoid(z);
  const error = output - target;
  const dOutput = error * sigmoidDerivative(z);
  weight -= lr * dOutput * input;
  bias -= lr * dOutput;
}

const finalOutput = forward(input);
console.log(`finalOutput: ${finalOutput.toFixed(4)}`);
console.log(`weight: ${weight.toFixed(4)}`);
console.log(`bias: ${bias.toFixed(4)}`);
