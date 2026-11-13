import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { readFile } from 'node:fs/promises';
import { loadDataset } from './dataset.js';
import { summarize } from './domain.js';
import { writeResult } from './output.js';
import { resolvePolicy } from './reco/config.js';
import { loadModel, recommend, trainModel } from './reco/model.js';
import { evaluate } from './reco/evaluate.js';

const COMMANDS = {
  summary: { options: ['--input', '--out'], defaults: { input: 'data/records.json' } },
  inspect: { options: ['--input', '--out'], defaults: { input: 'data/records.json' } },
  train: {
    options: ['--input', '--policy', '--model', '--metrics'],
    defaults: { input: 'data/records.json', policy: 'data/policy.json', model: 'var/model.json', metrics: 'var/metrics.json' }
  },
  recommend: { options: ['--model', '--user', '--k', '--at'], defaults: { model: 'var/model.json' } }
};

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new Error(`参数 ${name} 应为整数`);
  return parsed;
}

export function parseArguments(args) {
  const [command = 'summary', ...rest] = args;
  const spec = COMMANDS[command];
  if (!spec) throw new Error(`命令应为 ${Object.keys(COMMANDS).join(' 或 ')}`);
  const result = { command, ...spec.defaults };
  const seen = new Set();
  for (let index = 0; index < rest.length; index += 2) {
    const name = rest[index];
    const value = rest[index + 1];
    if (!spec.options.includes(name) || !value || value.startsWith('--') || seen.has(name)) {
      throw new Error(`参数应为 ${spec.options.join(' 路径 ')} 形式，不可重复或缺值`);
    }
    seen.add(name);
    result[name.slice(2)] = value;
  }
  if (command === 'recommend') {
    if (result.k !== undefined) result.k = parseInteger(result.k, '--k');
    if (result.at !== undefined) result.at = parseInteger(result.at, '--at');
  }
  return result;
}

async function trainCommand(options) {
  const dataset = await loadDataset(options.input);
  const policy = resolvePolicy(JSON.parse(await readFile(options.policy, 'utf8')));
  const model = trainModel(dataset.records, policy, dataset.fingerprint);
  const metrics = evaluate(dataset.records, policy, model);
  await writeResult(options.model, model);
  await writeResult(options.metrics, metrics);
  return {
    model: options.model,
    metrics: options.metrics,
    inputFingerprint: dataset.fingerprint,
    window: metrics.window,
    trainedRecords: model.trainedRecords,
    mappedUsers: model.users.length,
    mappedItems: model.items.length,
    population: metrics.population,
    modelOverall: metrics.systems.model.overall,
    baselineOverall: metrics.systems.baseline.overall,
    delta: metrics.delta,
    noRelevantUsers: metrics.noRelevantUsers.map(entry => entry.user)
  };
}

export async function main(args) {
  const options = parseArguments(args);
  if (options.command === 'train') return trainCommand(options);
  if (options.command === 'recommend') {
    const model = await loadModel(options.model);
    return recommend(model, options);
  }
  const dataset = await loadDataset(options.input);
  const result = options.command === 'summary' ? summarize(dataset.records)
    : { records: dataset.records.length, fingerprint: dataset.fingerprint,
      fields: dataset.records.length ? Object.keys(dataset.records[0]).sort() : [] };
  const document = { ...result, inputFingerprint: dataset.fingerprint };
  if (options.out) await writeResult(options.out, document);
  return document;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).then(result => {
    process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  }).catch(error => {
    process.stderr.write(`处理失败：${error.message}\n`);
    process.exitCode = 1;
  });
}
