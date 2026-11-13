import { main } from '../src/cli.js';
import { readFile } from 'node:fs/promises';

const started = performance.now();
console.log('读取本地演练记录并核对输入指纹');
console.log(JSON.stringify(await main(['inspect']), null, 2));
console.log('现有领域核算');
console.log(JSON.stringify(await main(['summary','--out','var/summary.json']), null, 2));
console.log('业务配置');
console.log(JSON.stringify(JSON.parse(await readFile('data/policy.json', 'utf8')), null, 2));
console.log('训练隐式反馈推荐器并与非个性化热门基线离线比较（模型与指标写入 var/）');
console.log(JSON.stringify(await main(['train','--model','var/model.json','--metrics','var/metrics.json']), null, 2));
console.log('加载模型复核推荐：活跃用户（隐因子打分）');
console.log(JSON.stringify(await main(['recommend','--model','var/model.json','--user','u0','--k','5','--at','31']), null, 2));
console.log('加载模型复核推荐：未知用户（可解释热门兜底，下架商品说明原因）');
console.log(JSON.stringify(await main(['recommend','--model','var/model.json','--user','u99','--k','3','--at','34']), null, 2));
console.log(`完成，用时 ${Math.ceil(performance.now() - started)} 毫秒`);
