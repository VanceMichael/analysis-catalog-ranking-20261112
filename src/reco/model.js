// 模型训练、落盘校验与在线打分路径。recommend 命令与离线评估共用同一套
// 打分逻辑（rankFor），保证“训练一次、重复复核”时口径一致。
import { readFile } from 'node:fs/promises';
import { aggregateTrain, availableAt, buildAvailability, splitRecords } from './split.js';
import { mulberry32, trainALS } from './als.js';

export const MODEL_FORMAT = 'implicit-als';
export const MODEL_VERSION = 1;

export const OBJECTIVE = '置信加权隐式 ALS：已曝光(用户,商品)对按互动强度加权回归；'
  + '已曝光未互动对以 exposureConfidence/exposurePreference 软参与；'
  + '未曝光对不当作负例，仅以 unobservedConfidence 均匀先验锚定（不随热度加权，可置 0）；无负采样';

const byId = (left, right) => left.localeCompare(right, 'en');

export function trainModel(records, policy, inputFingerprint) {
  const { train } = splitRecords(records, policy.trainBefore, policy.validationBefore);
  const aggregated = aggregateTrain(train, policy.kindWeights);
  const users = [...aggregated.users.keys()].sort(byId);
  const items = [...aggregated.items.keys()].sort(byId);
  const userIndex = new Map(users.map((user, index) => [user, index]));
  const itemIndex = new Map(items.map((item, index) => [item, index]));

  const entries = [];
  for (const pair of aggregated.pairs.values()) {
    const positive = pair.strength > 0;
    const confidence = positive ? 1 + policy.alpha * pair.strength
      : pair.exposed ? policy.exposureConfidence : 0;
    if (confidence === 0) continue;
    entries.push({
      u: userIndex.get(pair.user),
      i: itemIndex.get(pair.item),
      c: confidence,
      p: positive ? 1 : policy.exposurePreference
    });
  }
  const { userFactors, itemFactors } = trainALS(
    entries, users.length, items.length,
    { ...policy, base: policy.unobservedConfidence }, mulberry32(policy.seed));

  const popularity = items
    .map(item => ({ item, score: aggregated.items.get(item).interactionStrength }))
    .sort((left, right) => right.score - left.score || byId(left.item, right.item));
  const availability = Object.fromEntries([...buildAvailability(records).entries()]
    .sort(([left], [right]) => byId(left, right)));
  const trainInteraction = Object.fromEntries(users
    .map(user => [user, aggregated.users.get(user).interactionStrength]));

  return {
    format: MODEL_FORMAT,
    version: MODEL_VERSION,
    objective: OBJECTIVE,
    inputFingerprint,
    config: policy,
    window: { trainBefore: policy.trainBefore, validationBefore: policy.validationBefore },
    trainedRecords: train.length,
    knownUntil: records.length ? Math.max(...records.map(record => record.time)) : policy.trainBefore,
    users,
    items,
    userFactors,
    itemFactors,
    trainInteraction,
    popularity,
    availability
  };
}

// 加载时严格校验结构，版本不支持直接拒绝，不做静默迁移。
export function validateModel(model) {
  if (model === null || typeof model !== 'object' || Array.isArray(model)) throw new Error('模型应为对象');
  if (model.format !== MODEL_FORMAT) throw new Error('不支持的模型格式');
  if (model.version !== MODEL_VERSION) throw new Error('不支持的模型版本');
  const rank = model.config?.rank;
  if (!Number.isSafeInteger(rank) || rank < 1) throw new Error('模型缺少有效的 config.rank');
  for (const field of ['users', 'items', 'userFactors', 'itemFactors']) {
    if (!Array.isArray(model[field])) throw new Error(`模型缺少数组字段 ${field}`);
  }
  if (model.userFactors.length !== model.users.length) throw new Error('模型用户映射与因子行数不一致');
  if (model.itemFactors.length !== model.items.length) throw new Error('模型商品映射与因子行数不一致');
  for (const [field, rows] of [['userFactors', model.userFactors], ['itemFactors', model.itemFactors]]) {
    for (const row of rows) {
      if (!Array.isArray(row) || row.length !== rank || !row.every(Number.isFinite)) {
        throw new Error(`模型 ${field} 维度应为 ${rank} 且数值有限`);
      }
    }
  }
  if (!Array.isArray(model.popularity) || model.popularity.length !== model.items.length) {
    throw new Error('模型热门兜底表与商品映射不一致');
  }
  if (model.availability === null || typeof model.availability !== 'object') {
    throw new Error('模型缺少可售时间线');
  }
  if (!Number.isSafeInteger(model.window?.trainBefore) || !Number.isSafeInteger(model.knownUntil)) {
    throw new Error('模型缺少训练时间界限');
  }
  return model;
}

export async function loadModel(path) {
  return validateModel(JSON.parse(await readFile(path, 'utf8')));
}

// 打分核心：已知且训练期有正向互动的用户用隐因子点积（mf）；
// 其余用户用训练期热门兜底（popularity-fallback），并给出可解释原因。
// 热门兜底分数来自训练窗口，不读取验证或更晚的互动记录。
export function rankFor(model, user, candidates) {
  const userPosition = model.users.indexOf(user);
  const popularityScore = new Map(model.popularity.map(entry => [entry.item, entry.score]));
  const active = userPosition >= 0 && (model.trainInteraction[user] ?? 0) > 0;
  let source;
  let note;
  if (active) {
    source = 'mf';
  } else {
    source = 'popularity-fallback';
    note = userPosition < 0
      ? '用户不在训练映射中，使用训练期热门兜底'
      : '用户训练期无正向互动，使用训练期热门兜底';
  }
  const scored = candidates.map(item => {
    let score;
    if (active) {
      const itemPosition = model.items.indexOf(item);
      const userVector = model.userFactors[userPosition];
      const itemVector = model.itemFactors[itemPosition];
      score = userVector.reduce((sum, value, f) => sum + value * itemVector[f], 0);
    } else {
      score = popularityScore.get(item) ?? 0;
    }
    return { item, score, source };
  });
  scored.sort((left, right) => right.score - left.score || byId(left.item, right.item));
  return { source, note, ranked: scored };
}

// 单次推荐：候选限于请求时刻可售且模型可评分的商品；
// 不可售商品不静默丢弃，列入 excluded 并说明原因。
export function recommend(model, { user, k, at }) {
  const limit = k ?? model.config.topK;
  const time = at ?? model.knownUntil;
  const candidates = [];
  const excluded = [];
  for (const item of model.items) {
    if (availableAt(model.availability[item], time)) candidates.push(item);
    else excluded.push({ item, reason: `在 t=${time} 不可售（已下架或尚未上架）` });
  }
  const { source, note, ranked } = rankFor(model, user, candidates);
  return {
    user: user ?? null,
    at: time,
    k: limit,
    source,
    ...(note ? { note } : {}),
    items: ranked.slice(0, limit).map((entry, index) => ({ rank: index + 1, ...entry })),
    excluded
  };
}
