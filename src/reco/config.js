// 推荐器业务配置：合并默认值并做严格校验，未知字段直接拒绝以避免静默拼写错误。
export const DEFAULT_POLICY = {
  // 时间界限：训练窗口为 time < trainBefore，验证窗口为 [trainBefore, validationBefore)。
  trainBefore: 30,
  validationBefore: 45,
  // 隐因子维度、推荐列表长度、确定性初始化种子。
  rank: 3,
  topK: 5,
  seed: 997,
  // ALS 迭代轮数与 L2 正则。
  iterations: 12,
  regularization: 0.1,
  // 置信权重：c = 1 + alpha * 加权互动强度；重复点击与不同强度互动由此放大置信。
  alpha: 1,
  // 各记录类型的强度权重；exposure 默认为 0（曝光本身不计入互动强度），可调大为弱正例。
  kindWeights: { exposure: 0, click: 1, purchase: 2 },
  // 已曝光未互动对的软约束：置信 exposureConfidence、偏好目标 exposurePreference。
  // 未曝光对不当作负例：仅以 unobservedConfidence 的微小均匀先验锚定（不随商品热度
  // 加权，可置 0 完全关闭），避免未曝光商品得分无约束地浮高；本推荐器不做负采样。
  exposureConfidence: 0.5,
  exposurePreference: 0,
  unobservedConfidence: 0.05,
  // 头部商品：按训练期热度降序累计覆盖该比例互动强度的最小商品集合，其余为长尾。
  longTailShare: 0.8
};

const INTEGER_FIELDS = ['trainBefore', 'validationBefore', 'rank', 'topK', 'seed', 'iterations'];
const NUMBER_FIELDS = ['regularization', 'alpha', 'exposureConfidence', 'exposurePreference', 'unobservedConfidence', 'longTailShare'];

export function resolvePolicy(raw = {}) {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('配置应为对象');
  }
  const known = new Set([...INTEGER_FIELDS, ...NUMBER_FIELDS, 'kindWeights']);
  for (const key of Object.keys(raw)) {
    if (!known.has(key)) throw new Error(`配置含未知字段 ${key}`);
  }
  const policy = { ...DEFAULT_POLICY, ...raw, kindWeights: { ...DEFAULT_POLICY.kindWeights, ...(raw.kindWeights ?? {}) } };
  for (const field of INTEGER_FIELDS) {
    if (!Number.isSafeInteger(policy[field])) throw new Error(`配置 ${field} 应为安全整数`);
  }
  for (const field of NUMBER_FIELDS) {
    if (typeof policy[field] !== 'number' || !Number.isFinite(policy[field])) {
      throw new Error(`配置 ${field} 应为有限数值`);
    }
  }
  if (policy.trainBefore >= policy.validationBefore) throw new Error('配置 trainBefore 应小于 validationBefore');
  if (policy.rank < 1 || policy.topK < 1 || policy.iterations < 1) throw new Error('配置 rank/topK/iterations 应至少为 1');
  if (policy.regularization <= 0) throw new Error('配置 regularization 应大于 0');
  if (policy.alpha < 0 || policy.exposureConfidence < 0 || policy.unobservedConfidence < 0) {
    throw new Error('配置 alpha/exposureConfidence/unobservedConfidence 不应为负');
  }
  if (policy.longTailShare <= 0 || policy.longTailShare >= 1) throw new Error('配置 longTailShare 应在 (0,1) 内');
  if (policy.kindWeights === null || typeof policy.kindWeights !== 'object' || Array.isArray(policy.kindWeights)) {
    throw new Error('配置 kindWeights 应为对象');
  }
  for (const [kind, weight] of Object.entries(policy.kindWeights)) {
    if (typeof weight !== 'number' || !Number.isFinite(weight) || weight < 0) {
      throw new Error(`配置 kindWeights.${kind} 应为非负有限数值`);
    }
  }
  return policy;
}
