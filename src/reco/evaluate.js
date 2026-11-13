// 离线复验：同一批验证用户、同一候选约束下，比较隐式反馈模型与非个性化热门基线。
// 指标按活跃用户/冷启动用户切片，并单列单次曝光、全新、全零互动 cohort 与
// 头部/长尾/新商品切片；没有相关商品的验证用户显式入账，不从分母里静默剔除。
import { aggregateTrain, availableAt, splitRecords } from './split.js';
import { rankFor } from './model.js';

const byId = (left, right) => left.localeCompare(right, 'en');

function dcg(hits) {
  return hits.reduce((sum, hit, index) => sum + (hit ? 1 / Math.log2(index + 2) : 0), 0);
}

function idcg(relevantCount, k) {
  return dcg(new Array(Math.min(relevantCount, k)).fill(true));
}

// 用户切片：active 与 cold-start 互斥；其余为可叠加的 cohort 标记。
function userFlags(userStat) {
  const flags = [];
  if (userStat.records === 0) flags.push('new');
  if (userStat.records === 1 && userStat.exposures === 1) flags.push('singleExposure');
  if (userStat.interactionRecords > 0 && userStat.interactionStrength === 0) flags.push('zeroInteraction');
  return flags;
}

// 商品切片：new 为训练窗口无记录（训练后才上架）；head 为按训练期热度降序、
// 累计覆盖 longTailShare 互动强度的最小集合；其余训练期商品为 longTail。
function itemSlices(aggregated, allItems, longTailShare) {
  const slices = new Map();
  const total = [...aggregated.items.values()].reduce((sum, stat) => sum + stat.interactionStrength, 0);
  const ranked = [...aggregated.items.entries()]
    .sort((left, right) => right[1].interactionStrength - left[1].interactionStrength || byId(left[0], right[0]));
  let covered = 0;
  const head = new Set();
  for (const [item, stat] of ranked) {
    if (covered < longTailShare * total) {
      head.add(item);
      covered += stat.interactionStrength;
    }
  }
  for (const item of allItems) {
    slices.set(item, !aggregated.items.has(item) ? 'new' : head.has(item) ? 'head' : 'longTail');
  }
  return { slices, head };
}

function rankBaseline(model, candidates) {
  const score = new Map(model.popularity.map(entry => [entry.item, entry.score]));
  return candidates
    .map(item => ({ item, score: score.get(item) ?? 0, source: 'popularity-baseline' }))
    .sort((left, right) => right.score - left.score || byId(left.item, right.item));
}

function scoreList(ranked, relevant, k) {
  const top = ranked.slice(0, k);
  const hits = top.map(entry => relevant.has(entry.item));
  return {
    recommendations: top.map((entry, index) => ({ rank: index + 1, ...entry, hit: hits[index] })),
    recall: relevant.size ? hits.filter(Boolean).length / relevant.size : 0,
    ndcg: relevant.size ? dcg(hits) / idcg(relevant.size, k) : 0
  };
}

function aggregate(results, key, universe, popularity, head) {
  const evaluated = results.length;
  const mean = selector => evaluated
    ? results.reduce((sum, result) => sum + selector(result), 0) / evaluated : 0;
  const recommended = results.flatMap(result => result[key].recommendations.map(entry => entry.item));
  return {
    users: evaluated,
    recallAtK: mean(result => result[key].recall),
    ndcgAtK: mean(result => result[key].ndcg),
    coverage: universe.size ? new Set(recommended).size / universe.size : 0,
    popularityBias: {
      meanTrainPopularity: recommended.length
        ? recommended.reduce((sum, item) => sum + (popularity.get(item) ?? 0), 0) / recommended.length : 0,
      headShare: recommended.length
        ? recommended.filter(item => head.has(item)).length / recommended.length : 0
    }
  };
}

function itemSliceRecall(results, key, slices) {
  const tally = new Map();
  for (const result of results) {
    const recommended = new Set(result[key].recommendations.map(entry => entry.item));
    for (const item of result.relevant) {
      const slice = slices.get(item);
      if (!tally.has(slice)) tally.set(slice, { relevant: 0, hits: 0 });
      const entry = tally.get(slice);
      entry.relevant += 1;
      if (recommended.has(item)) entry.hits += 1;
    }
  }
  return Object.fromEntries([...tally.entries()].sort(([left], [right]) => byId(left, right))
    .map(([slice, entry]) => [slice, { ...entry, recallAtK: entry.relevant ? entry.hits / entry.relevant : 0 }]));
}

export function evaluate(records, policy, model) {
  const { train, validation } = splitRecords(records, policy.trainBefore, policy.validationBefore);
  const aggregated = aggregateTrain(train, policy.kindWeights);
  const k = policy.topK;

  const validationByUser = new Map();
  for (const record of validation) {
    if (!validationByUser.has(record.user)) validationByUser.set(record.user, []);
    validationByUser.get(record.user).push(record);
  }
  const allItems = [...new Set(records.map(record => record.item))].sort(byId);
  const { slices, head } = itemSlices(aggregated, allItems, policy.longTailShare);
  const popularity = new Map(model.popularity.map(entry => [entry.item, entry.score]));

  const results = [];
  const noRelevantUsers = [];
  for (const user of [...validationByUser.keys()].sort(byId)) {
    const trail = [...validationByUser.get(user)]
      .sort((left, right) => left.time - right.time || byId(left.id, right.id));
    const userStat = aggregated.users.get(user)
      ?? { records: 0, exposures: 0, interactionRecords: 0, interactionStrength: 0 };
    const slice = userStat.interactionStrength > 0 ? 'active' : 'cold-start';
    const flags = userFlags(userStat);
    const requestTime = trail[0].time;
    const relevant = new Set();
    let sawInteraction = false;
    for (const record of trail) {
      if (record.kind === 'exposure') continue;
      if (record.strength > 0) relevant.add(record.item);
      else sawInteraction = true;
    }
    // 没有相关商品的验证用户：显式入账并说明去向，不进入指标分母也不删除。
    if (relevant.size === 0) {
      noRelevantUsers.push({
        user, slice, flags, validationRecords: trail.length,
        reason: sawInteraction
          ? '验证窗口内互动强度均为 0，无可核对的相关商品'
          : '验证窗口内只有曝光记录，无可核对的相关商品'
      });
      continue;
    }
    const candidates = model.items.filter(item => availableAt(model.availability[item], requestTime));
    const excluded = model.items.filter(item => !candidates.includes(item))
      .map(item => ({ item, reason: `请求时刻 t=${requestTime} 不可售` }));
    const unreachableRelevant = [...relevant].filter(item => !candidates.includes(item))
      .sort(byId)
      .map(item => ({
        item,
        reason: model.items.includes(item) ? '请求时刻不可售' : '训练窗口内无记录，模型无因子（训练后上架）'
      }));
    const modelRun = rankFor(model, user, candidates);
    results.push({
      user, slice, flags, requestTime,
      relevant: [...relevant].sort(byId),
      candidates,
      candidateCount: candidates.length,
      excluded,
      unreachableRelevant,
      model: { source: modelRun.source, ...(modelRun.note ? { note: modelRun.note } : {}), ...scoreList(modelRun.ranked, relevant, k) },
      baseline: scoreList(rankBaseline(model, candidates), relevant, k)
    });
  }

  const universe = new Set(results.flatMap(result => result.candidates));
  const systems = {};
  for (const key of ['model', 'baseline']) {
    const groups = { active: [], 'cold-start': [] };
    const cohorts = { new: [], singleExposure: [], zeroInteraction: [] };
    for (const result of results) {
      groups[result.slice].push(result);
      for (const flag of result.flags) cohorts[flag]?.push(result);
    }
    systems[key] = {
      overall: aggregate(results, key, universe, popularity, head),
      userSlices: Object.fromEntries(Object.entries(groups)
        .map(([name, group]) => [name, aggregate(group, key, universe, popularity, head)])),
      cohorts: Object.fromEntries(Object.entries(cohorts)
        .map(([name, group]) => [name, aggregate(group, key, universe, popularity, head)])),
      itemSlices: itemSliceRecall(results, key, slices)
    };
  }

  return {
    inputFingerprint: model.inputFingerprint,
    window: { trainBefore: policy.trainBefore, validationBefore: policy.validationBefore },
    k,
    objective: model.objective,
    definitions: {
      候选空间: '请求时刻可售（按该时刻之前最近的可售标记判定）且模型可评分的商品；逐用户记录在 users[].candidates',
      相关商品: '验证窗口内强度大于 0 的互动（点击/购买）商品；曝光不计入相关',
      recallAtK: '命中相关商品数 / 该用户相关商品数，含不可评分相关商品（记为未命中），宏平均',
      ndcgAtK: '二值相关的折损累计增益 / 理想增益，宏平均',
      coverage: '被推荐过的商品数 / 被评估用户候选空间并集',
      popularityBias: '推荐商品的训练期热度均值（meanTrainPopularity）与头部商品占比（headShare）',
      userSlices: 'active=训练期互动强度>0；cold-start=其余（含全新、单次曝光、全零互动）',
      cohorts: 'new=训练窗口无记录；singleExposure=训练窗口恰好一条曝光；zeroInteraction=有互动记录但强度全为 0',
      itemSlices: 'head=按训练期热度覆盖 longTailShare 的最小集合；longTail=其余训练期商品；new=训练后才上架',
      noRelevantUsers: '验证窗口内无可核对相关商品的用户，不计入指标分母，逐人列出去向'
    },
    population: {
      validationUsers: validationByUser.size,
      evaluatedUsers: results.length,
      noRelevantUsers: noRelevantUsers.length
    },
    systems,
    delta: {
      recallAtK: systems.model.overall.recallAtK - systems.baseline.overall.recallAtK,
      ndcgAtK: systems.model.overall.ndcgAtK - systems.baseline.overall.ndcgAtK,
      coverage: systems.model.overall.coverage - systems.baseline.overall.coverage,
      headShare: systems.model.overall.popularityBias.headShare - systems.baseline.overall.popularityBias.headShare
    },
    users: results,
    noRelevantUsers
  };
}
