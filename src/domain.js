import { validateRecords } from './dataset.js';

export function summarize(records) {
  validateRecords(records);
  const grouped = new Map();
  for (const row of [...records].sort((left, right) => left.id.localeCompare(right.id, 'en'))) {
    const key = String(row["kind"]);
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(row);
  }
  const groups = [...grouped.entries()].sort(([left], [right]) => left.localeCompare(right, 'en'))
    .map(([key, members]) => {
      const totals = Object.fromEntries(["time", "strength"].map(field => [field,
        members.reduce((sum, row) => sum + row[field], 0)]));
      return { key, count: members.length, sum: totals["strength"], items: new Set(members.map(row => row.item)).size };
    });
  return { groupBy: "kind", valueField: "strength", count: records.length, groups };
}
