// Loaded before the dashboard script; gain is shared with existing settings.
var gain = 1;
let talkWanted = false, talkRun = null, talkContext, talkNode, talkSource, talkMedia;
let talkGeneration = 0, talkInitializing = false;
function talkLog(message) {
  const host = document.querySelector('#elog');
  if (host) {
    const row = document.createElement('div');
    row.textContent = new Date().toLocaleTimeString() + '  ' + message;
    host.prepend(row);
    while (host.children.length > 100) host.lastChild.remove();
  }
}
async function talkFetch(url, options) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 1800);
  try {
    const response = await fetch(url, {...options, signal:controller.signal});
    if (!response.ok) throw Error((await response.json()).error || 'Talk HTTP '+response.status);
    return response;
  } finally { clearTimeout(timer); }
}
async function sendTalk(run) {
  if (run.sending) return;
  run.sending = true;
  try {
    while (run.queue.length) {
      const body = run.queue[0];
      const options = {method:'POST', body, headers:{
        'X-Talk-Session':run.id, 'X-Talk-Sequence':String(run.seq)}};
      // Retrying the same sequence is safe: server acknowledges duplicates.
      try { await talkFetch('/api/talk', options); }
      catch (_) { await talkFetch('/api/talk', options); }
      run.queue.shift();
      run.seq++;
    }
    if (run.ended) {
      await talkFetch('/api/talk/stop', {method:'POST', headers:{'X-Talk-Session':run.id}});
      if (talkRun === run) talkRun = null;
      talkLog('Talk finished');
    }
  } catch (e) {
    run.failed = true;
    run.queue.length = 0;
    talkWanted = false;
    talkNode?.port.postMessage('stop');
    document.querySelector('#talk').classList.remove('on');
    talkLog('Talk stopped: '+e.message+'. Check Wi-Fi / server load and try again.');
    try { await talkFetch('/api/talk/stop', {method:'POST',headers:{'X-Talk-Session':run.id}}); }
    catch (_) {}
    if (talkRun === run) talkRun = null;
  } finally { run.sending = false; }
}
async function talk(on) {
  if (on === talkWanted) return;
  talkWanted = on;
  const generation = ++talkGeneration;
  if (!on) {
    document.querySelector('#talk').classList.remove('on');
    if (talkRun) talkNode.port.postMessage('stop');
    return;
  }
  if (talkRun || talkInitializing) {
    talkWanted = false;
    talkLog('Wait for the previous Talk session to finish');
    return;
  }
  talkInitializing = true;
  try {
    if (!talkNode) {
      if (!navigator.mediaDevices || !window.AudioWorkletNode)
        throw Error('Talk requires HTTPS and a current browser with AudioWorklet');
      talkMedia = await navigator.mediaDevices.getUserMedia({
        audio:{echoCancellation:true, noiseSuppression:false, autoGainControl:false}});
      talkContext = new AudioContext();
      await talkContext.audioWorklet.addModule('/talk-worklet.js');
      talkNode = new AudioWorkletNode(talkContext, 'talk-capture');
      talkSource = talkContext.createMediaStreamSource(talkMedia);
      talkSource.connect(talkNode);
      talkNode.connect(talkContext.destination); // Worklet output is silent.
      talkNode.port.onmessage = ({data}) => {
        const run = talkRun;
        if (!run || run.failed) return;
        if (data === 'ended') { run.ended = true; sendTalk(run); return; }
        if (run.queue.length >= 8) {
          talkLog('Network cannot keep up. Stopping Talk instead of skipping words.');
          talk(false);
          return;
        }
        const pcm = new Int16Array(data.length);
        let peak = 0;
        for (let i=0;i<data.length;i++) {
          peak = Math.max(peak,Math.abs(data[i]));
          pcm[i] = Math.max(-1,Math.min(1,data[i]*gain))*32767;
        }
        document.querySelector('#mic').style.width = Math.min(100,peak*150)+'%';
        run.queue.push(pcm.buffer);
        sendTalk(run);
      };
    }
    await talkContext.resume();
    if (!talkWanted || generation !== talkGeneration) return;
    const response = await talkFetch('/api/talk/start',{method:'POST'});
    const {id} = await response.json();
    if (!talkWanted || generation !== talkGeneration) {
      await talkFetch('/api/talk/stop',{method:'POST',headers:{'X-Talk-Session':id}});
      return;
    }
    talkRun = {id, seq:0, queue:[], sending:false, ended:false, failed:false};
    talkNode.port.postMessage('start');
    document.querySelector('#talk').classList.add('on');
    talkLog('Talk transmitting');
  } catch(e) {
    talkWanted = false;
    if (!talkNode && talkMedia) talkMedia.getTracks().forEach(t=>t.stop());
    talkLog(e.message);
  } finally { talkInitializing = false; }
}
document.addEventListener('DOMContentLoaded', () => {
  const button = document.querySelector('#talk');
  button.addEventListener('pointerdown', e => {
    e.preventDefault(); button.setPointerCapture(e.pointerId); talk(true);
  });
  for (const name of ['pointerup','pointercancel','lostpointercapture'])
    button.addEventListener(name,()=>talk(false));
});
