// 置信加权隐式 ALS（Hu-Koren-Volinsky 思路）的确定性实现。
// 只在传入的 (用户, 商品) 对上回归：未曝光对不进入 entries，因而不被当作负例；
// 已曝光未互动对以调用方给定的软置信/软偏好参与。无负采样，结果只依赖种子与配置。

// mulberry32：小巧的确定性伪随机源，仅用于因子初始化。
export function mulberry32(seed) {
  let state = seed >>> 0;
  return function next() {
    state = (state + 0x6D2B79F5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

// 高斯消元（部分主元）解 n×n 线性方程组；输入不被改写。
export function solveLinearSystem(matrix, vector) {
  const n = vector.length;
  const augmented = matrix.map((row, index) => [...row, vector[index]]);
  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < n; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    if (augmented[column][column] === 0) throw new Error('线性方程组奇异，请增大 regularization');
    for (let row = 0; row < n; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column] / augmented[column][column];
      if (factor === 0) continue;
      for (let k = column; k <= n; k += 1) augmented[row][k] -= factor * augmented[column][k];
    }
  }
  return augmented.map((row, index) => row[n] / augmented[index][index]);
}

// entries: [{u, i, c, p}]，u/i 为从 0 开始的下标，c 为置信权重，p 为偏好目标。
// base 为未曝光对的均匀先验置信（unobservedConfidence）：A = λI + base·YᵀY + Σ_obs (c-base)·yyᵀ，
// 与 Hu-Koren-Volinsky 的稠密形式等价，但 base 远小于 1 且不随商品热度加权；base 为 0 时
// 未曝光对完全不参与。每轮先固定商品因子解全部用户，再固定用户因子解全部商品，结果确定。
export function trainALS(entries, userCount, itemCount, { rank, iterations, regularization, base = 0 }, random) {
  const byUser = Array.from({ length: userCount }, () => []);
  const byItem = Array.from({ length: itemCount }, () => []);
  for (const entry of entries) {
    byUser[entry.u].push(entry);
    byItem[entry.i].push(entry);
  }
  let itemFactors = Array.from({ length: itemCount },
    () => Array.from({ length: rank }, () => (random() - 0.5) * 0.2));
  let userFactors = Array.from({ length: userCount }, () => new Array(rank).fill(0));

  const gram = factors => {
    const result = Array.from({ length: rank }, () => new Array(rank).fill(0));
    for (const y of factors) {
      for (let f = 0; f < rank; f += 1) {
        for (let g = 0; g < rank; g += 1) result[f][g] += y[f] * y[g];
      }
    }
    return result;
  };

  const solve = (pairs, otherFactors, otherGram) => {
    const matrix = Array.from({ length: rank }, (_, f) => {
      const row = new Array(rank).fill(0);
      row[f] = regularization;
      return row;
    });
    for (let f = 0; f < rank; f += 1) {
      for (let g = 0; g < rank; g += 1) matrix[f][g] += base * otherGram[f][g];
    }
    const vector = new Array(rank).fill(0);
    for (const pair of pairs) {
      const y = otherFactors[pair.other];
      for (let f = 0; f < rank; f += 1) {
        vector[f] += pair.c * pair.p * y[f];
        for (let g = 0; g < rank; g += 1) matrix[f][g] += (pair.c - base) * y[f] * y[g];
      }
    }
    return solveLinearSystem(matrix, vector);
  };

  for (let round = 0; round < iterations; round += 1) {
    const itemGram = gram(itemFactors);
    userFactors = byUser.map(pairs => solve(pairs.map(e => ({ ...e, other: e.i })), itemFactors, itemGram));
    const userGram = gram(userFactors);
    itemFactors = byItem.map(pairs => solve(pairs.map(e => ({ ...e, other: e.u })), userFactors, userGram));
  }
  return { userFactors, itemFactors };
}
