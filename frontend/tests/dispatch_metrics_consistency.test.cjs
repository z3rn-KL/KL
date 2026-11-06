// Pure selector test: uses built-in Node assertions, and transpiles the actual TS module.
// Run after npm install: node tests/dispatch_metrics_consistency.test.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const filename = path.join(__dirname, '../src/lib/dispatchMetrics.ts');
const js = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText;
const moduleExports = {};
vm.runInNewContext(js, { exports: moduleExports, require }, { filename });
const {newOrdersOnDay, carryoverEnteringDay, workloadForDay, outstandingAtEndOfDay, deliveredOnDay, pendingAssignedWeightKg, ordersBelongingToShipper, deliveredSlaCounts, activeShippersOnDay} = moduleExports;
const orders = [
  {order_id:'a',release_day:1,delivery_status:'DELIVERED',delivered_day:1,sla_status:'ON_TIME',weight_kg:2,assigned_shipper_id:'s1'},
  {order_id:'b',release_day:1,delivery_status:'DELIVERED',delivered_day:2,sla_status:'OVERDUE',weight_kg:3,assigned_shipper_id:'s1'},
  {order_id:'c',release_day:1,delivery_status:'CARRIED_OVER',delivered_day:null,sla_status:'AT_RISK',weight_kg:4,assigned_shipper_id:null},
  {order_id:'d',release_day:2,delivery_status:'ASSIGNED',delivered_day:null,sla_status:'AT_RISK',weight_kg:6,assigned_shipper_id:'s2'},
];
assert.equal(newOrdersOnDay(orders,1),3);
assert.equal(newOrdersOnDay(orders,2),1);
assert.equal(carryoverEnteringDay(orders,1),0);
assert.equal(carryoverEnteringDay(orders,2),2); // b delivered on Day 2, c still pending
assert.equal(workloadForDay(orders,2),3);
assert.equal(outstandingAtEndOfDay(orders,1).map(o=>o.order_id).join(','),'b,c');
assert.equal(deliveredOnDay(orders,1),1);
assert.equal(deliveredOnDay(orders,2),1);
assert.equal(pendingAssignedWeightKg(orders),6);
assert.equal(deliveredSlaCounts(orders).ratePct,50);
assert.equal(deliveredSlaCounts([]).ratePct,null);
assert.equal(ordersBelongingToShipper(orders,{shipper_id:'s1',assigned_order_ids:[],completed_order_ids:['a','b']}).length,2);
assert.equal(activeShippersOnDay(orders, [{shipper_id:'s1', assigned_order_ids:[], completed_order_ids:['a','b']}, {shipper_id:'s2', assigned_order_ids:['d'], completed_order_ids:[]}], 1), 1);
assert.equal(activeShippersOnDay(orders, [{shipper_id:'s1', assigned_order_ids:[], completed_order_ids:['a','b']}, {shipper_id:'s2', assigned_order_ids:['d'], completed_order_ids:[]}], 2), 2);
assert.equal(activeShippersOnDay([{order_id:'a',release_day:1,delivery_status:'DELIVERED',delivered_day:1,assigned_shipper_id:'s1'}], [{shipper_id:'s1',assigned_order_ids:[],completed_order_ids:['a']}], 2), 0);
// Three-day example: Day 1 delivers 84/85, Day 2 receives 85 new + 1 carryover.
const scenario = [];
for (let i=0;i<85;i++) scenario.push({order_id:`d1_${i}`,release_day:1,delivered_day:i<84?1:2,delivery_status:'DELIVERED'});
for (let i=0;i<85;i++) scenario.push({order_id:`d2_${i}`,release_day:2,delivered_day:2,delivery_status:'DELIVERED'});
assert.equal(newOrdersOnDay(scenario,2),85);
assert.equal(carryoverEnteringDay(scenario,2),1);
assert.equal(workloadForDay(scenario,2),86); // Backend incorrectly reports 170 here
assert.equal(outstandingAtEndOfDay(scenario,1).length,1);
assert.equal(workloadForDay(scenario,3),0);
console.log('Dispatch metrics: day-scoped selectors + carryover scenario PASS');

// Scientific contract: frontend's feature labels MUST equal actual Python encoder order.
const rlLabelFile = path.join(__dirname, '../src/lib/rlFeatureLabels.ts');
const rlJS = ts.transpileModule(fs.readFileSync(rlLabelFile, 'utf8'), {
  compilerOptions: {target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS},
}).outputText;
const rlExports = {};
vm.runInNewContext(rlJS, {exports: rlExports, require}, {filename: rlLabelFile});
const pythonEncoderText = fs.readFileSync(path.join(__dirname, '../../python/rl/proposed/candidate_encoder.py'), 'utf8');
const pythonFeatureNamesMatch = pythonEncoderText.match(/FEATURE_NAMES\s*=\s*\(([\s\S]*?)\)/);
assert.ok(pythonFeatureNamesMatch, 'Python CandidateEncoder.FEATURE_NAMES declaration must exist');
const encoderKeys = [...pythonFeatureNamesMatch[1].matchAll(/"([^"]+)"/g)].map(x=>x[1]);
const frontendKeys = rlExports.formattedRlFeatures([1,2,3,4,5,6,7,8]).map(x=>x.key);
assert.equal(frontendKeys.join(','), encoderKeys.join(','), 'React labels must match actual CandidateEncoder feature vector');
console.log('RL feature vector matches Python CandidateEncoder order');
