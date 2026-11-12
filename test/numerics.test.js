import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { summarize } from '../src/domain.js';

const seed = JSON.parse(await readFile('data/records.json', 'utf8')).records[0];
const probe = patches => summarize(patches.map((patch, index) => ({...seed, id: `p${index}`, ...patch}))).groups[0];

test('互动商品覆盖与互动强度分开核算', () => {

  assert.equal(probe([{item: 'a', strength: 20}, {item: 'a', strength: 1}, {item: 'b', strength: 2}]).items, 2);
  assert.equal(probe([{item: 'a', strength: 0}]).items, 1);
});
