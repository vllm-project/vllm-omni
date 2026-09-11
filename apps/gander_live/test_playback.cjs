const fs=require('fs'),vm=require('vm'),assert=require('assert');
let Processor;
const events=[];
vm.runInNewContext(fs.readFileSync(__dirname+'/app/static/playback_worklet.js','utf8'),{AudioWorkletProcessor:class {constructor(){this.port={postMessage:e=>events.push(e)}}},sampleRate:24000,registerProcessor:(_,c)=>Processor=c});
const p=new Processor();
function send(m){p.port.onmessage({data:m})}
send({type:'audio',responseId:'a',pcm:new Int16Array(24000).fill(8192)});
send({type:'drain',responseId:'a'});
send({type:'audio',responseId:'b',pcm:new Int16Array(12000).fill(16384)});
send({type:'drain',responseId:'b'});
let peak=0;
for(let i=0;i<300;i++){let o=new Float32Array(128);p.process([],[[o]]);peak=Math.max(peak,...o);}
assert.equal(peak,0.5);
assert.deepEqual(events.filter(e=>e.type==='playback-drained').map(e=>[e.responseId,e.playedMs]),[['a',1000],['b',500]]);
send({type:'audio',responseId:'old',pcm:new Int16Array(24000).fill(30000)});
send({type:'clear'});
let o=new Float32Array(128);p.process([],[[o]]);assert(o.every(v=>v===0));
console.log('PASS response-specific playback durations, PCM scaling, cancellation clear');
const q=new Processor();
q.port.onmessage({data:{type:'audio',responseId:'old',pcm:new Int16Array(128).fill(8192)}});
q.port.onmessage({data:{type:'audio',responseId:'new',pcm:new Int16Array(128).fill(16384)}});
q.port.onmessage({data:{type:'cancel',responseId:'old'}});
q.port.onmessage({data:{type:'drain',responseId:'new'}});
let n=new Float32Array(128);q.process([],[[n]]);assert(n.every(v=>v===0.5));
console.log('PASS cancelling old response preserves newer queued audio');
const gapper=new Processor();
const inject=m=>gapper.port.onmessage({data:m});
const render=()=>gapper.process([],[[new Float32Array(128)]]);
inject({type:'audio',responseId:'gap',pcm:new Int16Array(128).fill(4096)});
gapper.waitFrames=0;
render();
for(let i=0;i<15;i++)render(); // 80ms internal wait
inject({type:'audio',responseId:'gap',pcm:new Int16Array(128).fill(4096)});
render();
for(let i=0;i<30;i++)render(); // final wait must not inflate internal gaps
inject({type:'drain',responseId:'gap'});render();
assert.equal(events.find(e=>e.type==='playback-drained'&&e.responseId==='gap').underrunMs,80);
console.log('PASS measures internal underruns without counting trailing silence');
