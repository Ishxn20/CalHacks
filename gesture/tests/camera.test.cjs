const {test}=require('node:test');
const assert=require('node:assert/strict');
const Camera=require('../web/camera.js');
function stream(name) {
  const track={name,stopped:false,stop(){this.stopped=true;}};
  return {track,getTracks:()=>[track]};
}
function deferred() {
  let resolve,reject;
  const promise=new Promise((ok,fail)=>{resolve=ok;reject=fail;});
  return {promise,resolve,reject};
}
const failure=name=>Object.assign(new Error('Original browser error'),{name});

test('selecting an iPhone respects that device, with no forced FaceTime or resolution',async()=>{
  let requested;const iphone=stream('iPhone');
  const c=new Camera({getUserMedia:async constraints=>{requested=constraints;return iphone;}});
  assert.equal(await c.open('iphone-id','Ishan Camera'),iphone);
  assert.deepEqual(requested,{video:{deviceId:{exact:'iphone-id'}},audio:false});
});

test('a late permission response cannot replace the newly selected camera or leak a stream',async()=>{
  const a=deferred(),b=deferred(),sa=stream('old'),sb=stream('new');
  const c=new Camera({getUserMedia:({video})=>video.deviceId.exact==='a'?a.promise:b.promise});
  const first=c.open('a'),second=c.open('b');
  b.resolve(sb);assert.equal(await second,sb);
  a.resolve(sa);assert.equal(await first,null);
  assert.equal(c.stream,sb);assert.equal(sa.track.stopped,true);assert.equal(sb.track.stopped,false);
});

test('a late failed request cannot clear a newer working capture',async()=>{
  const old=deferred(),working=stream('iPhone');let n=0;
  const c=new Camera({getUserMedia:()=>++n===1?old.promise:Promise.resolve(working)});
  const first=c.open('a');await c.open('b');
  old.reject(failure('NotAllowedError'));
  assert.equal(await first,null);assert.equal(c.stream,working);assert.equal(working.track.stopped,false);
});

test('switching releases the previous camera before requesting another one',async()=>{
  const old=stream('old'),next=stream('iPhone');let calls=0;
  const c=new Camera({getUserMedia:async()=>{
    if(++calls===1)return old;
    assert.equal(old.track.stopped,true);return next;
  }});
  await c.open('old');await c.open('iphone');assert.equal(c.stream,next);
});

test('closing the page while permission is pending releases a late capture',async()=>{
  const wait=deferred(),s=stream('late');
  const c=new Camera({getUserMedia:()=>wait.promise});
  const opening=c.open();c.stop();wait.resolve(s);
  assert.equal(await opening,null);assert.equal(c.stream,null);assert.equal(s.track.stopped,true);
});

test('a camera still releasing its old session is retried once',async()=>{
  let calls=0,waited=0;const s=stream('iPhone');
  const c=new Camera({getUserMedia:async()=>{if(++calls===1)throw failure('NotReadableError');return s;}},async ms=>{waited=ms;});
  assert.equal(await c.open('iphone'),s);assert.equal(calls,2);assert.equal(waited,600);
});

test('permission errors retain their real cause and never trigger retries or fallback',async()=>{
  let calls=0;const e=failure('NotAllowedError');
  const c=new Camera({getUserMedia:async()=>{calls++;throw e;}});
  await assert.rejects(c.open('iphone'),err=>err===e);assert.equal(calls,1);
});

test('a refreshed Continuity Camera ID is found by the selected label',async()=>{
  const ids=[];const s=stream('iPhone');
  const c=new Camera({
    getUserMedia:async({video})=>{ids.push(video.deviceId.exact);if(ids.length===1)throw failure('OverconstrainedError');return s;},
    enumerateDevices:async()=>[{kind:'videoinput',deviceId:'new-id',label:'Ishan Camera'}],
  });
  assert.equal(await c.open('old-id','Ishan Camera'),s);assert.deepEqual(ids,['old-id','new-id']);
});

test('a missing iPhone does not silently switch to the Mac camera',async()=>{
  const e=failure('NotFoundError');let calls=0;
  const c=new Camera({getUserMedia:async()=>{calls++;throw e;},
    enumerateDevices:async()=>[{kind:'videoinput',deviceId:'facetime',label:'FaceTime HD Camera'}]});
  await assert.rejects(c.open('iphone','Ishan Camera'),err=>err===e);assert.equal(calls,1);
});
