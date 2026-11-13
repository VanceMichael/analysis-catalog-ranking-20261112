// 时间切分、可售时间线与训练期聚合。所有函数只读取传入记录，不隐式取全集。

// 按时间界限切分：训练窗口 time < trainBefore，验证窗口 [trainBefore, validationBefore)。
// 验证窗口之外的记录（更晚的未来）不参与训练也不参与评估。
export function splitRecords(records, trainBefore, validationBefore) {
  const train = [];
  const validation = [];
  for (const record of records) {
    if (record.time < trainBefore) train.push(record);
    else if (record.time < validationBefore) validation.push(record);
  }
  return { train, validation };
}

// 每个商品的可售状态时间线：[[time, available], ...] 升序、折叠连续重复状态。
// available 是记录时刻的可售标记，属于目录状态而非用户行为；
// 服务时按“请求时刻之前最近一条记录”判定，不读取该时刻之后的记录。
export function buildAvailability(records) {
  const sorted = [...records].sort((left, right) =>
    left.time - right.time || left.id.localeCompare(right.id, 'en'));
  const timelines = new Map();
  for (const record of sorted) {
    const timeline = timelines.get(record.item) ?? [];
    if (timeline.length === 0 || timeline[timeline.length - 1][1] !== record.available) {
      timeline.push([record.time, record.available]);
    }
    timelines.set(record.item, timeline);
  }
  return timelines;
}

// 商品在指定时刻是否可售；从未出现过的商品视为尚未上架。
export function availableAt(timeline, time) {
  let state = false;
  for (const [changedAt, available] of timeline ?? []) {
    if (changedAt > time) break;
    state = available;
  }
  return state;
}

// 单条记录对互动强度的贡献：权重来自可配置的 kindWeights，
// exposure 默认权重 0（曝光不计入互动），未列出的互动类型默认权重 1。
export function interactionWeight(record, kindWeights) {
  if (record.kind === 'exposure') return (kindWeights.exposure ?? 0) * record.strength;
  return (kindWeights[record.kind] ?? 1) * record.strength;
}

// 训练窗口聚合。只产出“出现过”的 (用户, 商品) 对：曝光或互动过才进入，
// 从未曝光的对不生成记录，因此不会被当作负例。
export function aggregateTrain(trainRecords, kindWeights) {
  const pairs = new Map();
  const users = new Map();
  const items = new Map();
  const ensure = (map, key, init) => {
    if (!map.has(key)) map.set(key, init());
    return map.get(key);
  };
  for (const record of trainRecords) {
    const userStat = ensure(users, record.user, () => ({ records: 0, exposures: 0, interactionRecords: 0, interactionStrength: 0 }));
    const itemStat = ensure(items, record.item, () => ({ records: 0, interactionStrength: 0 }));
    const pair = ensure(pairs, JSON.stringify([record.user, record.item]), () => ({ user: record.user, item: record.item, exposed: false, strength: 0 }));
    const weight = interactionWeight(record, kindWeights);
    userStat.records += 1;
    itemStat.records += 1;
    if (record.kind === 'exposure') {
      userStat.exposures += 1;
      pair.exposed = true;
    } else {
      userStat.interactionRecords += 1;
    }
    pair.strength += weight;
    userStat.interactionStrength += weight;
    itemStat.interactionStrength += weight;
  }
  return { pairs, users, items };
}
