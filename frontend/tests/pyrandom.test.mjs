import assert from "node:assert/strict";
import test from "node:test";
import { execFileSync } from "node:child_process";
import { PythonRandom } from "../js/pyrandom.js";

test("PythonRandom seed42 matches known Python random.random outputs", () => {
  const rng = new PythonRandom(42);
  assert.deepEqual(Array.from({ length: 10 }, () => rng.random()), [
    0.6394267984578837, 0.025010755222666936, 0.27502931836911926,
    0.22321073814882275, 0.7364712141640124, 0.6766994874229113,
    0.8921795677048454, 0.08693883262941615, 0.4219218196852704, 0.029797219438070344,
  ]);
});

test("all painting seeds match Python across multiple MT twist boundaries", () => {
  const seeds = [0, 42, 142, 192, 242, 342, 442, 542, 592, 642, 742, 842, 892, 1042, 2042, 3042];
  const expected = JSON.parse(execFileSync("python", ["-c",
    "import json,random,sys; seeds=json.loads(sys.argv[1]); result=[]" +
    "\nfor seed in seeds:\n r=random.Random(seed); result.append([r.random() for _ in range(1000)])" +
    "\nprint(json.dumps(result))", JSON.stringify(seeds)], { encoding: "utf8" }));
  for (let i = 0; i < seeds.length; i++) {
    const rng = new PythonRandom(seeds[i]);
    assert.deepEqual(Array.from({ length: 1000 }, () => rng.random()), expected[i], `seed ${seeds[i]}`);
  }
});

test("uniform and reseeding preserve deterministic streams", () => {
  const rng = new PythonRandom(42);
  assert.equal(rng.uniform(-180, 180), -180 + 360 * 0.6394267984578837);
  rng.seed(42);
  assert.equal(rng.random(), 0.6394267984578837);
  assert.equal(new PythonRandom(-42).random(), new PythonRandom(42).random());
});
