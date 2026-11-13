import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { loadDataset } from '../src/dataset.js';
import { writeResult } from '../src/output.js';
import { resolvePolicy } from '../src/reco/config.js';
import { aggregateTrain, availableAt, buildAvailability, splitRecords } from '../src/reco/split.js';
import { mulberry32, solveLinearSystem, trainALS } from '../src/reco/als.js';
import { loadModel, recommend, trainModel, validateModel } from '../src/reco/model.js';
import { evaluate } from '../src/reco/evaluate.js';

const dataset = await loadDataset('data/records.json');
const policy = resolvePolicy(JSON.parse(await readFile('data/policy.json', 'utf8')));
const model = trainModel(dataset.records, policy, dataset.fingerprint);
const metrics = evaluate(dataset.records, policy, model);

test('配置合并默认值并拒绝未知字段与非法取值', () => {
  const defaults = resolvePolicy();
  assert.equal(defaults.rank, 3);
  assert.equal(resolvePolicy({ rank: 4 }).rank, 4);
  assert.throws(() => resolvePolicy({ rank: 0 }), /rank/);
  assert.throws(() => resolvePolicy({ trainBefore: 10, validationBefore: 10 }), /trainBefore/);
  assert.throws(() => resolvePolicy({ typo: 1 }), /未知字段/);
  assert.throws(() => resolvePolicy({ kindWeights: { click: -1 } }), /kindWeights/);
  assert.throws(() => resolvePolicy({ unobservedConfidence: -0.1 }), /不应为负/);
  assert.throws(() => resolvePolicy({ longTailShare: 1 }), /longTailShare/);
});

test('训练与验证按时间切分，窗口外记录两侧都不进入', () => {
  const { train, validation } = splitRecords(dataset.records, 30, 45);
  assert.ok(train.every(record => record.time < 30));
  assert.ok(validation.every(record => record.time >= 30 && record.time < 45));
  assert.equal(train.length + validation.length, dataset.records.length);
});

test('可售时间线按请求时刻判定，未上架与下架都有明确状态', () => {
  const timelines = buildAvailability(dataset.records);
  assert.equal(availableAt(timelines.get('i7'), 31), true);
  assert.equal(availableAt(timelines.get('i7'), 32), false);
  assert.equal(availableAt(timelines.get('i8'), 31), false);
  assert.equal(availableAt(timelines.get('i8'), 32), true);
  assert.equal(availableAt(timelines.get('不存在'), 10), false);
});

test('未曝光对不进入训练对，重复点击与类型权重提升置信', () => {
  const weights = { exposure: 0, click: 1, purchase: 2 };
  const once = [
    { id: 'a', user: 'u', item: 'i', time: 0, kind: 'exposure', strength: 1, available: true },
    { id: 'b', user: 'u', item: 'i', time: 1, kind: 'click', strength: 2, available: true },
    { id: 'c', user: 'u', item: 'j', time: 0, kind: 'exposure', strength: 1, available: true }
  ];
  const twice = [...once, { id: 'd', user: 'u', item: 'i', time: 2, kind: 'click', strength: 2, available: true }];
  const pairOf = records => aggregateTrain(records, weights).pairs.get(JSON.stringify(['u', 'i'])).strength;
  assert.ok(pairOf(twice) > pairOf(once));
  const aggregated = aggregateTrain(once, weights);
  assert.equal(aggregated.pairs.has(JSON.stringify(['u', '从未曝光'])), false);
  assert.equal(aggregated.pairs.get(JSON.stringify(['u', 'j'])).strength, 0);
  assert.equal(aggregated.pairs.get(JSON.stringify(['u', 'j'])).exposed, true);
  const heavier = aggregateTrain(once, { ...weights, click: 3 });
  assert.ok(heavier.pairs.get(JSON.stringify(['u', 'i'])).strength > aggregated.pairs.get(JSON.stringify(['u', 'i'])).strength);
});

test('线性求解与 ALS 训练是确定性的', () => {
  assert.deepEqual(solveLinearSystem([[2, 0], [0, 4]], [2, 8]), [1, 2]);
  const entries = [{ u: 0, i: 0, c: 2, p: 1 }, { u: 0, i: 1, c: 0.5, p: 0 }, { u: 1, i: 1, c: 3, p: 1 }];
  const options = { rank: 2, iterations: 5, regularization: 0.1, base: 0.05 };
  const first = trainALS(entries, 2, 2, options, mulberry32(7));
  const second = trainALS(entries, 2, 2, options, mulberry32(7));
  assert.deepEqual(first, second);
  const fit = first.userFactors[0].reduce((sum, v, f) => sum + v * first.itemFactors[0][f], 0);
  assert.ok(fit > 0.5, `正例拟合应明显大于 0，实际 ${fit}`);
});

test('训练不读取未来记录：截断数据训练得到相同映射、因子与兜底表', () => {
  const truncated = dataset.records.filter(record => record.time < policy.trainBefore);
  const early = trainModel(truncated, policy, dataset.fingerprint);
  assert.deepEqual(model.users, early.users);
  assert.deepEqual(model.items, early.items);
  assert.deepEqual(model.userFactors, early.userFactors);
  assert.deepEqual(model.itemFactors, early.itemFactors);
  assert.deepEqual(model.popularity, early.popularity);
  assert.deepEqual(model.trainInteraction, early.trainInteraction);
});

test('模型落盘加载后对相同请求给出一致推荐', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'model-'));
  try {
    const path = join(folder, 'model.json');
    await writeResult(path, model);
    const loaded = await loadModel(path);
    assert.deepEqual(loaded, model);
    for (const request of [
      { user: 'u0', k: 5, at: 31 },
      { user: 'u99', k: 3, at: 34 },
      { user: 'u13', k: 4, at: 33 },
      { user: undefined, k: 5, at: 30 }
    ]) {
      assert.deepEqual(recommend(loaded, request), recommend(model, request));
      assert.deepEqual(recommend(loaded, request), recommend(loaded, request));
    }
  } finally { await rm(folder, { recursive: true, force: true }); }
});

test('未知用户与零互动用户走可解释兜底，下架商品说明原因', () => {
  const unknown = recommend(model, { user: 'u99', k: 3, at: 34 });
  assert.equal(unknown.source, 'popularity-fallback');
  assert.match(unknown.note, /不在训练映射/);
  assert.ok(unknown.items.every(entry => entry.source === 'popularity-fallback'));
  const zero = recommend(model, { user: 'u13', k: 3, at: 31 });
  assert.equal(zero.source, 'popularity-fallback');
  assert.match(zero.note, /无正向互动/);
  const delisted = unknown.excluded.find(entry => entry.item === 'i7');
  assert.match(delisted.reason, /不可售/);
  const early = recommend(model, { user: 'u0', k: 8, at: 2 });
  assert.ok(early.excluded.some(entry => entry.item === 'i4'));
  const active = recommend(model, { user: 'u0', k: 5, at: 31 });
  assert.equal(active.source, 'mf');
  assert.ok(active.items.every(entry => entry.source === 'mf'));
});

test('模型文件校验拒绝损坏内容与不支持版本', () => {
  assert.throws(() => validateModel({ ...model, version: 2 }), /版本/);
  assert.throws(() => validateModel({ ...model, userFactors: model.userFactors.slice(1) }), /不一致/);
  assert.throws(() => validateModel({ ...model, popularity: [] }), /兜底/);
});

test('评估覆盖要求的切片与去向，候选限于当时可售商品', () => {
  assert.equal(metrics.population.validationUsers, 6);
  assert.equal(metrics.population.evaluatedUsers + metrics.population.noRelevantUsers, 6);
  const gone = metrics.noRelevantUsers.find(entry => entry.user === 'u14');
  assert.match(gone.reason, /无可核对的相关商品/);
  const byUser = new Map(metrics.users.map(entry => [entry.user, entry]));
  assert.deepEqual(byUser.get('u10').flags, ['new']);
  assert.deepEqual(byUser.get('u12').flags, ['singleExposure']);
  assert.deepEqual(byUser.get('u13').flags, ['zeroInteraction']);
  assert.equal(byUser.get('u0').slice, 'active');
  const u10 = byUser.get('u10');
  assert.deepEqual(u10.unreachableRelevant, [
    { item: 'i8', reason: '训练窗口内无记录，模型无因子（训练后上架）' }]);
  assert.ok(!byUser.get('u11').candidates.includes('i7'));
  assert.ok(byUser.get('u11').excluded.some(entry => entry.item === 'i7'));
  for (const entry of metrics.users) {
    for (const rec of entry.model.recommendations) assert.ok(['mf', 'popularity-fallback'].includes(rec.source));
  }
  assert.equal(metrics.systems.model.itemSlices.new.relevant, 1);
  assert.equal(metrics.systems.model.itemSlices.new.recallAtK, 0);
  assert.deepEqual(evaluate(dataset.records, policy, model), metrics);
});

test('指标数值可手工复核：兜底排序下 recall 与 NDCG 符合定义', () => {
  const toy = {
    format: 'implicit-als', version: 1, objective: '', inputFingerprint: '',
    config: policy, window: { trainBefore: 30, validationBefore: 45 },
    trainedRecords: 0, knownUntil: 40,
    users: [], items: ['i0', 'i1'],
    userFactors: [], itemFactors: [],
    trainInteraction: {}, popularity: [{ item: 'i1', score: 5 }, { item: 'i0', score: 1 }],
    availability: { i0: [[0, true]], i1: [[0, true]] }
  };
  const records = [
    { id: 'v1', user: 'u', item: 'i0', time: 31, kind: 'click', strength: 2, available: true },
    { id: 'v2', user: 'u', item: 'i1', time: 31, kind: 'exposure', strength: 1, available: true }
  ];
  const toyPolicy = resolvePolicy({ topK: 1 });
  const result = evaluate(records, toyPolicy, toy);
  assert.equal(result.users[0].model.recall, 0);
  assert.equal(result.users[0].model.ndcg, 0);
  const wide = evaluate(records, resolvePolicy({ topK: 2 }), toy);
  assert.equal(wide.users[0].model.recall, 1);
  assert.equal(wide.users[0].model.ndcg, 1 / Math.log2(3));
});

test('命令行 train 落盘模型与指标，recommend 可重复复核', async () => {
  const folder = await mkdtemp(join(tmpdir(), 'reco-cli-'));
  try {
    const modelPath = join(folder, 'model.json');
    const metricsPath = join(folder, 'metrics.json');
    const train = spawnSync(process.execPath, ['src/cli.js', 'train', '--model', modelPath, '--metrics', metricsPath], { encoding: 'utf8' });
    assert.equal(train.status, 0, train.stderr);
    const summary = JSON.parse(train.stdout);
    assert.equal(summary.population.evaluatedUsers, 5);
    const onDisk = JSON.parse(await readFile(metricsPath, 'utf8'));
    assert.ok(onDisk.systems.model.overall.recallAtK >= 0);
    const ask = ['recommend', '--model', modelPath, '--user', 'u0', '--k', '5', '--at', '31'];
    const first = spawnSync(process.execPath, ['src/cli.js', ...ask], { encoding: 'utf8' });
    const second = spawnSync(process.execPath, ['src/cli.js', ...ask], { encoding: 'utf8' });
    assert.equal(first.status, 0, first.stderr);
    assert.equal(first.stdout, second.stdout);
    assert.equal(JSON.parse(first.stdout).source, 'mf');
    const missing = spawnSync(process.execPath, ['src/cli.js', 'recommend', '--model', join(folder, '无.json')], { encoding: 'utf8' });
    assert.equal(missing.status, 1);
    const badK = spawnSync(process.execPath, ['src/cli.js', 'recommend', '--model', modelPath, '--k', 'x'], { encoding: 'utf8' });
    assert.equal(badK.status, 1);
  } finally { await rm(folder, { recursive: true, force: true }); }
});
